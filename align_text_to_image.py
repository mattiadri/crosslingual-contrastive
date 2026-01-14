#!/usr/bin/env python3
"""
Align multilingual text embeddings to the image embedding space using images as pivot.

Outputs under --out_dir:
- alignment/W_<train_split>_<langs>.pt
- alignment/report_<train_split>_<langs>.json  (when --save_report)

Expected inputs (precomputed):
- embeddings_image/<split>/*.pt  (image embeddings)
- embeddings_text/<split>/*.pt   (text embeddings per language)

python align_text_to_image.py \
  --out_dir /home/mattia/crosslingual-contrastive/webdataset \
  --train_split full --eval_splits "" \
  --langs ar,de,en,es,fr,it,ja,pt,zh \
  --epochs 20 --steps_per_epoch 12 --batch_per_lang 32 \
  --lr 3e-3 --wd 5e-4 --temp 0.09 --warmup 90 \
  --prox_id 1e-3 --ortho 1e-4 --max_grad_norm 1.0 \
  --device cuda --save_report
"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, KFold

# ---------------------- I/O ----------------------

def load_image_embeds(split_dir: str) -> Tuple[Dict[str, int], torch.Tensor]:
    pt_paths = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
    if not pt_paths:
        raise SystemExit(f"No image embeddings found in {split_dir}")
    keys_all: List[str] = []
    embs_all: List[torch.Tensor] = []
    for p in pt_paths:
        payload = torch.load(p, map_location="cpu")
        E = payload["embeddings"].float()
        K = payload["keys"]
        keys = [str(k) for k in K]
        assert E.shape[0] == len(keys)
        embs_all.append(E)
        keys_all.extend(keys)
    I = torch.cat(embs_all, dim=0)
    key_to_idx = {k: i for i, k in enumerate(keys_all)}
    return key_to_idx, F.normalize(I, dim=-1)


def load_text_embeds(split_dir: str, langs: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
    pt_paths = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
    if not pt_paths:
        raise SystemExit(f"No text embeddings found in {split_dir}")
    out: Dict[str, Dict[str, torch.Tensor]] = {l: {"keys": [], "emb": []} for l in langs}
    for p in pt_paths:
        payload = torch.load(p, map_location="cpu")
        emb_by_lang = payload["embeddings"]
        keys_by_lang = payload["keys"]
        for l in langs:
            E = emb_by_lang.get(l)
            K = keys_by_lang.get(l)
            if E is None or K is None or len(K) == 0:
                continue
            out[l]["emb"].append(E.float())
            out[l]["keys"].extend([str(k) for k in K])
    for l in langs:
        out[l]["emb"] = torch.cat(out[l]["emb"], dim=0) if out[l]["emb"] else torch.empty(0, 1)
    return out


def build_aligned_arrays(img_map: Dict[str, int], I: torch.Tensor, T: Dict[str, Dict[str, torch.Tensor]], langs: List[str]):
    sets = [set(T[l]["keys"]) for l in langs]
    common = set(img_map.keys()).intersection(*sets)
    if not common:
        raise SystemExit("No common keys across images and requested languages.")
    common = sorted(common)
    idx_img = torch.tensor([img_map[k] for k in common], dtype=torch.long)
    I_shared = I[idx_img]
    T_list = []
    for l in langs:
        first_idx = {}
        for i, k in enumerate(T[l]["keys"]):
            if k not in first_idx:
                first_idx[k] = i
        rows = [first_idx[k] for k in common]
        X = T[l]["emb"][rows]
        T_list.append(F.normalize(X, dim=-1))
    return T_list, I_shared, common


# ---------------------- Retrieval metrics ----------------------

def retrieval_metrics(text: torch.Tensor, image: torch.Tensor, ks=(1, 5, 10)):
    sims = text @ image.T
    ranks = torch.argsort(sims, dim=1, descending=True)
    gt = torch.arange(text.size(0)).unsqueeze(1)
    out = {}
    for k in ks:
        out[f"R@{k}"] = (ranks[:, :k] == gt).any(dim=1).float().mean().item()
    inv = []
    for i in range(text.size(0)):
        rpos = (ranks[i] == i).nonzero(as_tuple=True)[0].item()
        inv.append(1.0 / (rpos + 1))
    out["mAP"] = float(np.mean(inv))
    return out


# ---------------------- Diagnostics ----------------------

def _sample_rows(N: int, max_n: int) -> np.ndarray:
    if N <= max_n:
        return np.arange(N)
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(N, size=max_n, replace=False))


def gram_corr_across_langs(X_list: List[torch.Tensor], sample: int = 2000) -> Dict[str, object]:
    """Pairwise Pearson correlation between flattened Gram matrices across languages."""
    if len(X_list) < 2:
        return {"mean": float("nan"), "matrix": []}
    n = X_list[0].size(0)
    idx = _sample_rows(n, sample)
    vecs = []
    for X in X_list:
        Xs = F.normalize(X[idx], dim=-1)
        G = (Xs @ Xs.T).cpu().numpy()
        iu = np.triu_indices_from(G, k=1)
        vecs.append(G[iu])
    L = len(vecs)
    mat = np.ones((L, L), dtype=float)
    for i in range(L):
        for j in range(i + 1, L):
            c = np.corrcoef(vecs[i], vecs[j])[0, 1]
            mat[i, j] = mat[j, i] = float(c)
    mean_offdiag = float(mat[np.triu_indices(L, k=1)].mean()) if L > 1 else float("nan")
    return {"mean": mean_offdiag, "matrix": mat.tolist()}


def effective_rank(X: torch.Tensor) -> float:
    """Effective rank via entropy of singular values."""
    Xc = X - X.mean(0, keepdim=True)
    _, S, _ = torch.linalg.svd(Xc, full_matrices=False)
    s = (S ** 2).cpu().numpy()
    s = s / (s.sum() + 1e-12)
    H = -(s * np.log(s + 1e-12)).sum()
    return float(np.exp(H))


def pca_components_for_var(X: torch.Tensor, var_thresh: float = 0.90) -> int:
    Xc = X - X.mean(0, keepdim=True)
    _, S, _ = torch.linalg.svd(Xc, full_matrices=False)
    s2 = (S ** 2).cpu().numpy()
    frac = np.cumsum(s2) / (s2.sum() + 1e-12)
    return int(1 + np.searchsorted(frac, var_thresh))


def mean_pairwise_cosine(X: torch.Tensor, sample: int = 2000) -> float:
    n = X.size(0)
    idx = _sample_rows(n, sample)
    Xs = F.normalize(X[idx], dim=-1)
    G = (Xs @ Xs.T).cpu().numpy()
    iu = np.triu_indices_from(G, k=1)
    return float(G[iu].mean())


def poz_and_entropy(X: torch.Tensor, eps: float = 1e-3, bins: int = 30) -> Tuple[float, float]:
    """Percentage of near-zero activations and average per-dim entropy."""
    Xn = X.cpu().numpy()
    poz = float((np.abs(Xn) < eps).mean())
    Xc = np.clip(Xn, -1.0, 1.0)
    ent = []
    for d in range(Xc.shape[1]):
        hist, _ = np.histogram(Xc[:, d], bins=bins, range=(-1, 1), density=True)
        p = hist / (hist.sum() + 1e-12)
        ent.append(float(-(p * np.log(p + 1e-12)).sum()))
    return poz, float(np.mean(ent))


def diagnostics_block(label: str, X_list: List[torch.Tensor], langs: List[str], sample: int = 2000, bins: int = 30) -> Dict[str, object]:
    gram = gram_corr_across_langs(X_list, sample=sample)
    per_lang = {}
    for l, X in zip(langs, X_list):
        er = effective_rank(X)
        pc90 = pca_components_for_var(X, var_thresh=0.90)
        iso = mean_pairwise_cosine(X, sample=sample)
        poz, ent = poz_and_entropy(X, eps=1e-3, bins=bins)
        per_lang[l] = {
            "effective_rank": round(er, 4),
            "pca90_components": int(pc90),
            "mean_pairwise_cosine": round(iso, 6),
            "PoZ": round(poz, 6),
            "entropy": round(ent, 6),
        }
    macro = {k: float(np.mean([per_lang[l][k] for l in per_lang])) for k in next(iter(per_lang.values())).keys()}
    return {
        "label": label,
        "gram_corr_mean": round(gram["mean"], 6),
        "gram_corr_matrix": gram["matrix"],
        "per_lang": per_lang,
        "macro_avg": {k: (round(v, 6) if not isinstance(v, int) else v) for k, v in macro.items()},
    }


# ---------------------- Report helpers ----------------------

def _round_metrics(d):
    return {k: {m: float(round(v, 6)) for m, v in d[k].items()} for k in d}

def _delta_metrics(before, after):
    out = {}
    for l in before.keys():
        out[l] = {m: float(round(after[l][m] - before[l][m], 6)) for m in before[l].keys()}
    return out

def _macro_avg(per_lang):
    keys = next(iter(per_lang.values())).keys()
    return {m: float(round(sum(per_lang[l][m] for l in per_lang) / len(per_lang), 6)) for m in keys}

def _result_block(split, kind, n_pairs, before, after):
    before = _round_metrics(before)
    after  = _round_metrics(after)
    delta  = _delta_metrics(before, after)
    block = {
        "kind": kind,
        "n_pairs": int(n_pairs),
        "per_lang": {"before": before, "after": after, "delta": delta},
        "macro_avg": {
            "before": _macro_avg(before),
            "after":  _macro_avg(after),
            "delta":  _macro_avg(delta)
        }
    }
    if split is not None:
        block["split"] = split
    return block


# ---------------------- Training utils ----------------------

def cosine_warmup(total_steps: int, base_lr: float, warmup: int):
    def get_lr(step):
        if step < warmup:
            return base_lr * (step + 1) / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return base_lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t)))
    return get_lr

def clip_loss(txt, img, temp=0.07):
    txt = F.normalize(txt, dim=-1)
    img = F.normalize(img, dim=-1)
    logits = (txt @ img.T) / temp
    y = torch.arange(txt.size(0), device=txt.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_split", default="full")
    ap.add_argument("--eval_splits", default="", help="Comma-separated list; empty = internal eval")
    ap.add_argument("--langs", default="en,it,es")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps_per_epoch", type=int, default=12)
    ap.add_argument("--batch_per_lang", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--kfold", type=int, default=1, help="If >1 and eval_splits is empty, run K-fold CV")
    ap.add_argument("--fold_id", type=int, default=-1, help="If >=0, run only this fold (0..k-1)")
    ap.add_argument("--save_report", action="store_true")
    # Regularization & stability
    ap.add_argument("--prox_id", type=float, default=0.0, help="L2 penalty on ΔW (equiv. ‖W−I‖²). 0 = off")
    ap.add_argument("--ortho", type=float, default=0.0, help="Soft orthogonality on (W^T W ≈ I). 0 = off")
    ap.add_argument("--max_grad_norm", type=float, default=0.0, help="Clip gradients by global norm; 0 = off")
    ap.add_argument("--early_stop_metric", default="R@1", choices=["R@1", "mAP"], help="Early stopping metric")
    # Diagnostics options
    ap.add_argument("--diag_sample", type=int, default=2000, help="Sample size for O(n^2) diagnostics")
    ap.add_argument("--diag_bins", type=int, default=30, help="Histogram bins for entropy")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]

    train_img_dir = os.path.join(args.out_dir, "embeddings_image", args.train_split)
    train_txt_dir = os.path.join(args.out_dir, "embeddings_text", args.train_split)
    if not os.path.isdir(train_img_dir):
        raise SystemExit(f"Missing image embeddings dir: {train_img_dir}")
    if not os.path.isdir(train_txt_dir):
        raise SystemExit(f"Missing text embeddings dir: {train_txt_dir}")

    img_map, I = load_image_embeds(train_img_dir)
    T = load_text_embeds(train_txt_dir, langs)
    T_list_all, I_shared_all, _ = build_aligned_arrays(img_map, I, T, langs)

    eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]

    d_img = I_shared_all.size(1)
    d_text = T_list_all[0].size(1)
    device = torch.device(args.device)

    save_dir = os.path.join(args.out_dir, "alignment")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    def run_one_train(
        T_list: List[torch.Tensor],
        I_shared: torch.Tensor,
        holdout_pair: Optional[Tuple[List[torch.Tensor], torch.Tensor]],
        run_tag: str,
    ):
        """Train ΔW. If holdout_pair is set, do early stopping on holdout and report holdout before/after macros."""
        N = I_shared.size(0)
        L = len(T_list)

        print(f"[{run_tag}] train items: {N} | langs={langs} | d_text={d_text} | d_img={d_img}")

        baseline_train = {langs[i]: retrieval_metrics(T_list[i], I_shared) for i in range(L)}
        diag_before = diagnostics_block("before", T_list, langs, sample=args.diag_sample, bins=args.diag_bins)

        # W = I + ΔW
        eye_rect = torch.zeros(d_text, d_img, device=device)
        for i in range(min(d_text, d_img)):
            eye_rect[i, i] = 1.0
        DeltaW = nn.Parameter(torch.zeros(d_text, d_img, device=device))
        opt = torch.optim.Adam([DeltaW], lr=args.lr, weight_decay=args.wd)

        eye_square = torch.eye(d_img, device=device)

        def W_eff():
            return eye_rect + DeltaW

        total_steps = args.epochs * args.steps_per_epoch
        lr_fn = cosine_warmup(total_steps, args.lr, args.warmup)

        def set_lr(step):
            for g in opt.param_groups:
                g["lr"] = lr_fn(step)

        I_dev = I_shared.to(device)
        T_dev = [t.to(device) for t in T_list]

        best_score = -1.0
        best_DeltaW: Optional[torch.Tensor] = None

        def _macro_score(metric_per_lang: dict, key: str) -> float:
            return float(np.mean([metric_per_lang[l][key] for l in metric_per_lang]))

        step = 0
        for epoch in range(args.epochs):
            for _ in range(args.steps_per_epoch):
                idx = torch.randint(0, N, (args.batch_per_lang,), device=device)
                batch_t = torch.cat([T_dev[i][idx] for i in range(L)], dim=0)
                batch_i = I_dev[idx].repeat(L, 1)

                We = W_eff()
                loss = clip_loss(batch_t @ We, batch_i, temp=args.temp)

                if args.prox_id > 0.0:
                    prox = torch.sum(DeltaW ** 2) / (d_text * d_img)
                    loss = loss + args.prox_id * prox

                if args.ortho > 0.0:
                    WT_W = We.t() @ We
                    ortho = torch.sum((WT_W - eye_square) ** 2) / (d_img * d_img)
                    loss = loss + args.ortho * ortho

                set_lr(step)
                opt.zero_grad()
                loss.backward()
                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_([DeltaW], args.max_grad_norm)
                opt.step()
                step += 1

            if holdout_pair is not None:
                T_val, I_val = holdout_pair
                with torch.no_grad():
                    We = W_eff()
                    T_proj_val = [F.normalize(t.to(device) @ We, dim=-1).cpu() for t in T_val]
                val_metrics = {langs[i]: retrieval_metrics(T_proj_val[i], I_val) for i in range(L)}
                score = _macro_score(val_metrics, args.early_stop_metric)
                if score > best_score:
                    best_score = score
                    best_DeltaW = DeltaW.detach().cpu().clone()
                    print(f"[{run_tag}][earlystop] epoch={epoch} best {args.early_stop_metric}={score:.4f}")

        if holdout_pair is not None and best_DeltaW is not None:
            DeltaW.data = best_DeltaW.to(device)

        with torch.no_grad():
            We = W_eff()
            T_proj_train = [F.normalize(t @ We, dim=-1).cpu() for t in T_dev]
        final_train = {langs[i]: retrieval_metrics(T_proj_train[i], I_shared) for i in range(L)}
        diag_after = diagnostics_block("after", T_proj_train, langs, sample=args.diag_sample, bins=args.diag_bins)

        def _delta_macro(d0, d1):
            out = {}
            for k in d0.keys():
                out[k] = float(round(d1[k] - d0[k], 6))
            return out

        config = {
            "out_dir": args.out_dir,
            "train_split": args.train_split,
            "langs": langs,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_per_lang": args.batch_per_lang,
            "lr": args.lr,
            "wd": args.wd,
            "temp": args.temp,
            "warmup": args.warmup,
            "device": args.device,
            "seed": args.seed,
            "version": "v5",
            "prox_id": args.prox_id,
            "ortho": args.ortho,
            "max_grad_norm": args.max_grad_norm,
            "early_stop_metric": args.early_stop_metric,
            "run_tag": run_tag,
        }

        report = {"version": "v5", "config": config, "results": [], "diagnostics": {}}
        report["results"].append(_result_block(None, "train", I_shared.size(0), baseline_train, final_train))
        report["diagnostics"]["train"] = {
            "before": diag_before,
            "after": diag_after,
            "delta_macro": _delta_macro(diag_before["macro_avg"], diag_after["macro_avg"]),
            "gram_corr_mean": {"before": diag_before["gram_corr_mean"], "after": diag_after["gram_corr_mean"]},
        }

        holdout_macro_before = None
        holdout_macro_after = None
        if holdout_pair is not None:
            T_val, I_val = holdout_pair
            base_h = {langs[i]: retrieval_metrics(T_val[i], I_val) for i in range(L)}
            with torch.no_grad():
                We_cpu = (eye_rect.detach().cpu() + DeltaW.detach().cpu())
                T_proj_h = [F.normalize(t @ We_cpu, dim=-1) for t in T_val]
            final_h = {langs[i]: retrieval_metrics(T_proj_h[i], I_val) for i in range(L)}
            report["results"].append(_result_block("holdout", "holdout", I_val.size(0), base_h, final_h))

            diag_h_before = diagnostics_block("before", T_val, langs, sample=args.diag_sample, bins=args.diag_bins)
            diag_h_after = diagnostics_block("after", T_proj_h, langs, sample=args.diag_sample, bins=args.diag_bins)
            report["diagnostics"]["holdout"] = {
                "before": diag_h_before,
                "after": diag_h_after,
                "delta_macro": _delta_macro(diag_h_before["macro_avg"], diag_h_after["macro_avg"]),
                "gram_corr_mean": {"before": diag_h_before["gram_corr_mean"], "after": diag_h_after["gram_corr_mean"]},
            }

            holdout_macro_before = report["results"][-1]["macro_avg"]["before"]
            holdout_macro_after  = report["results"][-1]["macro_avg"]["after"]

        We_cpu = (eye_rect.detach().cpu() + DeltaW.detach().cpu())
        return We_cpu, report, holdout_macro_before, holdout_macro_after

    # -------- K-fold CV (internal eval only) --------
    if not eval_splits and args.kfold and args.kfold > 1:
        N_all = I_shared_all.size(0)
        if N_all < args.kfold:
            raise SystemExit("Not enough paired samples for requested k-fold")
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

        fold_before = []
        fold_after = []
        folds_ran = 0

        for fold, (tr_idx, val_idx) in enumerate(kf.split(np.arange(N_all))):
            if args.fold_id >= 0 and fold != args.fold_id:
                continue

            tr_idx = torch.as_tensor(tr_idx, dtype=torch.long)
            val_idx = torch.as_tensor(val_idx, dtype=torch.long)

            I_tr, I_val = I_shared_all[tr_idx], I_shared_all[val_idx]
            T_tr = [t[tr_idx] for t in T_list_all]
            T_val = [t[val_idx] for t in T_list_all]

            run_tag = f"fold{fold}"
            We_cpu, rep, hold_b, hold_a = run_one_train(T_tr, I_tr, (T_val, I_val), run_tag=run_tag)

            w_path = os.path.join(save_dir, f"W_{args.train_split}_{'-'.join(langs)}_{run_tag}.pt")
            torch.save({"W": We_cpu, "langs": langs, "train_split": args.train_split, "run_tag": run_tag}, w_path)
            rep["config"]["weight_path"] = w_path
            rep_path = os.path.join(save_dir, f"report_{args.train_split}_{'-'.join(langs)}_{run_tag}.json")
            with open(rep_path, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)

            print(f"[{run_tag}] saved W -> {w_path}")
            print(f"[{run_tag}] saved report -> {rep_path}")

            folds_ran += 1
            if hold_b is not None and hold_a is not None:
                fold_before.append(hold_b)
                fold_after.append(hold_a)

        if fold_after:
            keys = list(fold_after[0].keys())

            mean_b = {k: float(np.mean([d[k] for d in fold_before])) for k in keys}
            mean_a = {k: float(np.mean([d[k] for d in fold_after])) for k in keys}

            # std with ddof=1 (sample std), consistent with paper "std over folds"
            std_b = {k: float(np.std([d[k] for d in fold_before], ddof=1)) for k in keys} if len(fold_before) > 1 else {k: 0.0 for k in keys}
            std_a = {k: float(np.std([d[k] for d in fold_after], ddof=1)) for k in keys} if len(fold_after) > 1 else {k: 0.0 for k in keys}

            delta_mean = {k: float(mean_a[k] - mean_b[k]) for k in keys}

            cv_summary = {
                "k": int(args.kfold),
                "n_folds_ran": int(folds_ran),
                "macro_before_mean": mean_b,
                "macro_before_std": std_b,
                "macro_after_mean": mean_a,
                "macro_after_std": std_a,
                "macro_delta_mean": delta_mean,
            }

            cv_path = os.path.join(save_dir, f"report_cv_{args.train_split}_{'-'.join(langs)}.json")
            with open(cv_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"version": "v5", "config": {"kfold": args.kfold, "seed": args.seed, "langs": langs, "train_split": args.train_split}, "cv": cv_summary},
                    f,
                    indent=2,
                )
            print(f"[cv] saved summary -> {cv_path}")

            # Canonical stats file for notebook/paper
            canon = {
                "version": "v1",
                "setup": {
                    "k_folds": int(args.kfold),
                    "folds_ran": int(folds_ran),
                    "split": "holdout",
                    "macro_avg": True,
                    "train_split": args.train_split,
                    "langs": langs,
                    "seed": int(args.seed),
                },
                "before": {k: {"mean": float(mean_b[k]), "std": float(std_b[k])} for k in keys},
                "after":  {k: {"mean": float(mean_a[k]), "std": float(std_a[k])} for k in keys},
                "delta":  {k: {"mean": float(delta_mean[k])} for k in keys},
            }
            canon_path = os.path.join(save_dir, f"cv_stats_holdout_macro_{args.train_split}_{'-'.join(langs)}.json")
            with open(canon_path, "w", encoding="utf-8") as f:
                json.dump(canon, f, indent=2)
            print(f"[cv] saved canonical stats -> {canon_path}")

        return

    # -------- Single run (original behavior) --------
    holdout_pair = None
    if not eval_splits:
        if not (0.0 < args.holdout < 1.0):
            raise SystemExit("--holdout must be in (0,1) when eval_splits is empty")
        N_all = I_shared_all.size(0)
        if N_all < 2:
            raise SystemExit("Not enough paired samples after alignment")
        idx = np.arange(N_all)
        idx_tr, idx_val = train_test_split(idx, test_size=args.holdout, random_state=args.seed, shuffle=True)
        idx_tr = torch.as_tensor(idx_tr, dtype=torch.long)
        idx_val = torch.as_tensor(idx_val, dtype=torch.long)
        I_tr, I_val = I_shared_all[idx_tr], I_shared_all[idx_val]
        T_tr = [t[idx_tr] for t in T_list_all]
        T_val = [t[idx_val] for t in T_list_all]
        holdout_pair = (T_val, I_val)
        T_list_run, I_run = T_tr, I_tr
    else:
        T_list_run, I_run = T_list_all, I_shared_all

    We_cpu, report, _, _ = run_one_train(T_list_run, I_run, holdout_pair, run_tag="single")

    w_path = os.path.join(save_dir, f"W_{args.train_split}_{'-'.join(langs)}.pt")
    torch.save({"W": We_cpu, "langs": langs, "train_split": args.train_split}, w_path)
    report["config"]["weight_path"] = w_path
    if not eval_splits:
        report["config"]["holdout"] = args.holdout
    else:
        report["config"]["eval_splits"] = eval_splits
    print(f"Saved W -> {w_path}")

    # External eval splits
    for es in eval_splits:
        img_dir = os.path.join(args.out_dir, "embeddings_image", es)
        txt_dir = os.path.join(args.out_dir, "embeddings_text", es)
        if not (os.path.isdir(img_dir) and os.path.isdir(txt_dir)):
            print(f"[eval:{es}] missing embeddings dirs (image or text), skipping.")
            continue
        img_map_e, I_e = load_image_embeds(img_dir)
        T_e = load_text_embeds(txt_dir, langs)
        try:
            T_list_e, I_shared_e, common_e = build_aligned_arrays(img_map_e, I_e, T_e, langs)
        except SystemExit as e:
            print(f"[eval:{es}] {e}. Skipping.")
            continue

        base_e = {langs[i]: retrieval_metrics(T_list_e[i], I_shared_e) for i in range(len(langs))}
        with torch.no_grad():
            T_proj_e = [F.normalize(t @ We_cpu, dim=-1) for t in T_list_e]
        final_e = {langs[i]: retrieval_metrics(T_proj_e[i], I_shared_e) for i in range(len(langs))}
        report["results"].append(_result_block(es, "eval", I_shared_e.size(0), base_e, final_e))

        def _delta_macro(d0, d1):
            out = {}
            for k in d0.keys():
                out[k] = float(round(d1[k] - d0[k], 6))
            return out

        diag_e_before = diagnostics_block("before", T_list_e, langs, sample=args.diag_sample, bins=args.diag_bins)
        diag_e_after = diagnostics_block("after", T_proj_e, langs, sample=args.diag_sample, bins=args.diag_bins)
        report["diagnostics"][f"eval:{es}"] = {
            "before": diag_e_before,
            "after": diag_e_after,
            "delta_macro": _delta_macro(diag_e_before["macro_avg"], diag_e_after["macro_avg"]),
            "gram_corr_mean": {"before": diag_e_before["gram_corr_mean"], "after": diag_e_after["gram_corr_mean"]},
        }
        print(f"[eval:{es}] done. common={len(common_e)}")

    if args.save_report:
        rep_path = os.path.join(save_dir, f"report_{args.train_split}_{'-'.join(langs)}.json")
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report -> {rep_path}")

if __name__ == "__main__":
    main()