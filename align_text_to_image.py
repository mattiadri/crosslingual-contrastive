#!/usr/bin/env python3
"""
Align multilingual text embeddings to the image embedding space using images as pivot.

Outputs under --out_dir/alignment:
- W_<train_split>_<langs>_<tag>_<head>.pt
- report_<train_split>_<langs>_<tag>_<head>.json  (when --save_report)

Expected inputs (precomputed):
- embeddings_image/<split>/*.pt  (image embeddings)
- embeddings_text/<split>/*.pt   (text embeddings per language)

#BEST from sweep:

python align_text_to_image.py \
  --out_dir webdataset --train_split full --eval_splits "" \
  --langs ar,de,en,es,fr,it,ja,pt,zh --device cuda --save_report \
  --epochs 20 --steps_per_epoch 12 --batch_per_lang 32 \
  --lr 1e-3 --wd 5e-4 --temp 0.07 --warmup 120 \
  --prox_id 1e-3 --ortho 0 --max_grad_norm 1.0 \
  --head linear --kfold 5 --seed 7

python align_text_to_image.py \
  --out_dir webdataset --train_split full --eval_splits "" \
  --langs ar,de,en,es,fr,it,ja,pt,zh --device cuda --save_report \
  --epochs 20 --steps_per_epoch 12 --batch_per_lang 32 \
  --lr 1e-3 --wd 5e-4 --temp 0.07 --warmup 120 \
  --prox_id 1e-3 --max_grad_norm 1.0 \
  --head mlp --mlp_hidden 512 --mlp_dropout 0.0 \
  --kfold 5 --seed 7

"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression


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
    out: Dict[str, Dict[str, Any]] = {l: {"keys": [], "emb": []} for l in langs}
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
    return out  # type: ignore


def build_aligned_arrays(
    img_map: Dict[str, int],
    I: torch.Tensor,
    T: Dict[str, Dict[str, torch.Tensor]],
    langs: List[str],
):
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

def retrieval_metrics(query: torch.Tensor, candidates: torch.Tensor, ks=(1, 5, 10)) -> Dict[str, float]:
    sims = query @ candidates.T
    ranks = torch.argsort(sims, dim=1, descending=True)
    gt = torch.arange(query.size(0)).unsqueeze(1)
    out: Dict[str, float] = {}
    for k in ks:
        out[f"R@{k}"] = (ranks[:, :k] == gt).any(dim=1).float().mean().item()
    inv = []
    for i in range(query.size(0)):
        rpos = (ranks[i] == i).nonzero(as_tuple=True)[0].item()
        inv.append(1.0 / (rpos + 1))
    out["MRR"] = float(np.mean(inv))
    return out


# ---------------------- Diagnostics ----------------------

def _sample_rows(N: int, max_n: int) -> np.ndarray:
    if N <= max_n:
        return np.arange(N)
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(N, size=max_n, replace=False))


def gram_corr_across_langs(X_list: List[torch.Tensor], sample: int = 2000) -> Dict[str, object]:
    if len(X_list) < 2:
        raise ValueError("Need at least 2 languages for gram_corr_across_langs")
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
    mean_offdiag = float(mat[np.triu_indices(L, k=1)].mean())
    return {"mean": mean_offdiag, "matrix": mat.tolist()}


def neighbor_overlap_across_langs(
    X_list: List[torch.Tensor],
    k: int = 10,
    sample: int = 2000,
) -> Dict[str, object]:
    if len(X_list) < 2:
        raise ValueError("Need at least 2 languages for neighbor_overlap_across_langs")
    if k <= 0:
        raise ValueError("k must be > 0 for neighbor_overlap_across_langs")

    n = X_list[0].size(0)
    if any(x.size(0) != n for x in X_list):
        raise ValueError("neighbor_overlap_across_langs requires aligned X_list with same n")

    idx = _sample_rows(n, sample)

    neigh = []
    for X in X_list:
        Xs = F.normalize(X[idx], dim=-1)
        sims = (Xs @ Xs.T).cpu().numpy()
        np.fill_diagonal(sims, -np.inf)
        kk = min(k, sims.shape[1] - 1)
        nn_idx = np.argpartition(-sims, kth=kk, axis=1)[:, :kk]
        row_scores = np.take_along_axis(sims, nn_idx, axis=1)
        order = np.argsort(-row_scores, axis=1)
        nn_idx = np.take_along_axis(nn_idx, order, axis=1)
        neigh.append(nn_idx)

    L = len(neigh)
    mat = np.ones((L, L), dtype=float)
    for a in range(L):
        for b in range(a + 1, L):
            ov = []
            for i in range(neigh[a].shape[0]):
                sa = set(map(int, neigh[a][i]))
                sb = set(map(int, neigh[b][i]))
                ov.append(len(sa & sb) / float(len(sa | sb) + 1e-12))
            mat[a, b] = mat[b, a] = float(np.mean(ov))
    mean_offdiag = float(mat[np.triu_indices(L, k=1)].mean())
    return {"mean": mean_offdiag, "matrix": mat.tolist(), "k": int(k), "sample_n": int(len(idx))}


def pivot_caption_retrieval_aligned(
    T_src: torch.Tensor,
    I_shared: torch.Tensor,
    T_tgt: torch.Tensor,
    ks=(1, 5, 10),
) -> Dict[str, float]:
    sims = T_src @ I_shared.T
    pred_img_idx = torch.argmax(sims, dim=1)
    I_pred = I_shared[pred_img_idx]
    return retrieval_metrics(I_pred, T_tgt, ks=ks)


def pivot_retrieval_block(
    T_list: List[torch.Tensor],
    I_shared: torch.Tensor,
    langs: List[str],
    ks=(1, 5, 10),
) -> Dict[str, object]:
    L = len(langs)
    per_pair: Dict[str, Dict[str, float]] = {}
    mat_r1 = np.full((L, L), np.nan, dtype=float)
    mat_mrr = np.full((L, L), np.nan, dtype=float)

    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            m = pivot_caption_retrieval_aligned(T_list[i], I_shared, T_list[j], ks=ks)
            per_pair[f"{langs[i]}->{langs[j]}"] = {k: float(round(v, 6)) for k, v in m.items()}
            mat_r1[i, j] = float(m["R@1"])
            mat_mrr[i, j] = float(m["MRR"])

    vals = list(per_pair.values())
    macro = {}
    for k in vals[0].keys():
        macro[k] = float(round(np.mean([v[k] for v in vals]), 6))

    return {
        "macro_avg": macro,
        "per_pair": per_pair,
        "matrix_R@1": mat_r1.tolist(),
        "matrix_MRR": mat_mrr.tolist(),
        "langs": langs,
    }


def effective_rank(X: torch.Tensor) -> float:
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
    Xn = X.cpu().numpy()
    poz = float((np.abs(Xn) < eps).mean())
    Xc = np.clip(Xn, -1.0, 1.0)
    ent = []
    for d in range(Xc.shape[1]):
        hist, _ = np.histogram(Xc[:, d], bins=bins, range=(-1, 1), density=True)
        p = hist / (hist.sum() + 1e-12)
        ent.append(float(-(p * np.log(p + 1e-12)).sum()))
    return poz, float(np.mean(ent))


def hubness_stats(X: torch.Tensor, k: int = 10, sample: int = 2000) -> Dict[str, float]:
    n = X.size(0)
    idx = _sample_rows(n, sample)
    Xs = F.normalize(X[idx], dim=-1)
    sims = (Xs @ Xs.T).cpu().numpy()
    np.fill_diagonal(sims, -np.inf)

    kk = min(k, sims.shape[1] - 1)
    nn_idx = np.argpartition(-sims, kth=kk, axis=1)[:, :kk]
    counts = np.bincount(nn_idx.reshape(-1), minlength=sims.shape[0]).astype(np.float64)

    mu = counts.mean()
    sigma = counts.std(ddof=0) + 1e-12
    skew = float(((counts - mu) ** 3).mean() / (sigma ** 3))

    m = len(counts)
    top = max(1, int(round(0.01 * m)))
    top_sum = float(np.sort(counts)[-top:].sum())
    total = float(counts.sum()) + 1e-12
    hub_ratio_1pct = float(top_sum / total)

    return {"hub_k": float(k), "hub_skew": skew, "hub_ratio_1pct": hub_ratio_1pct}


def language_id_probe(
    X_list: List[torch.Tensor],
    langs: List[str],
    sample_per_lang: int = 2000,
    test_size: float = 0.3,
    seed: int = 0,
    C: float = 1.0,
    max_iter: int = 2000,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    X_all = []
    y_all = []
    for li, X in enumerate(X_list):
        n = X.size(0)
        take = min(sample_per_lang, n)
        if take <= 1:
            raise ValueError("Not enough samples for language_id_probe")
        sel = rng.choice(n, size=take, replace=False)
        X_all.append(X[torch.as_tensor(sel)].cpu().numpy().astype(np.float32))
        y_all.append(np.full((take,), li, dtype=np.int64))

    Xmat = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    Xtr, Xte, ytr, yte = train_test_split(Xmat, y, test_size=test_size, random_state=seed, stratify=y)
    clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=max_iter, n_jobs=None)
    clf.fit(Xtr, ytr)
    acc = float((clf.predict(Xte) == yte).mean())
    return {"acc": acc, "n": float(len(y)), "k": float(len(np.unique(y)))}


def diagnostics_block(
    X_list: List[torch.Tensor],
    langs: List[str],
    sample: int,
    bins: int,
    hub_k: int,
    neigh_k: int,
    do_langid_probe: bool,
    langid_sample_per_lang: int,
    langid_test_size: float,
    langid_C: float,
    langid_seed: int,
) -> Dict[str, object]:
    gram = gram_corr_across_langs(X_list, sample=sample)
    neigh = neighbor_overlap_across_langs(X_list, k=neigh_k, sample=sample)

    per_lang = {}
    for l, X in zip(langs, X_list):
        er = effective_rank(X)
        pc90 = pca_components_for_var(X, var_thresh=0.90)
        iso = mean_pairwise_cosine(X, sample=sample)
        poz, ent = poz_and_entropy(X, eps=1e-3, bins=bins)
        hub = hubness_stats(X, k=hub_k, sample=sample)

        per_lang[l] = {
            "effective_rank": round(er, 4),
            "pca90_components": int(pc90),
            "mean_pairwise_cosine": round(iso, 6),
            "PoZ": round(poz, 6),
            "entropy": round(ent, 6),
            "hub_skew": round(float(hub["hub_skew"]), 6),
            "hub_ratio_1pct": round(float(hub["hub_ratio_1pct"]), 6),
        }

    macro = {k: float(np.mean([per_lang[l][k] for l in per_lang])) for k in next(iter(per_lang.values())).keys()}

    langid = None
    if do_langid_probe:
        langid = language_id_probe(
            X_list,
            langs,
            sample_per_lang=langid_sample_per_lang,
            test_size=langid_test_size,
            seed=langid_seed,
            C=langid_C,
        )

    out = {
        "macro_avg": {k: float(round(v, 6)) for k, v in macro.items()},
        "per_lang": per_lang,
        "gram_corr_mean": float(round(gram["mean"], 6)),
        "gram_corr_matrix": gram["matrix"],
        "neighbor_overlap_mean": float(round(neigh["mean"], 6)),
        "neighbor_overlap_matrix": neigh["matrix"],
        "neighbor_overlap_k": int(neigh["k"]),
    }
    if langid is not None:
        out["langid_probe"] = {"acc": float(round(langid["acc"], 6)), "n": int(langid["n"]), "k": int(langid["k"])}
    return out


# ---------------------- Report helpers ----------------------

def _round_metrics(d: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {k: {m: float(round(v, 6)) for m, v in d[k].items()} for k in d}


def _delta_metrics(before: Dict[str, Dict[str, float]], after: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for l in before.keys():
        out[l] = {m: float(round(after[l][m] - before[l][m], 6)) for m in before[l].keys()}
    return out


def _macro_avg(per_lang: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = next(iter(per_lang.values())).keys()
    return {m: float(round(sum(per_lang[l][m] for l in per_lang) / len(per_lang), 6)) for m in keys}


def _result_block(split: str, kind: str, n_pairs: int, before: Dict[str, Dict[str, float]], after: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    before = _round_metrics(before)
    after = _round_metrics(after)
    delta = _delta_metrics(before, after)
    return {
        "split": split,
        "kind": kind,
        "n_pairs": int(n_pairs),
        "per_lang": {"before": before, "after": after, "delta": delta},
        "macro_avg": {"before": _macro_avg(before), "after": _macro_avg(after), "delta": _macro_avg(delta)},
    }


def _mean_std(values: List[float]) -> Tuple[float, float]:
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return m, s


def _agg_mean_std_dict(list_of_dicts: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = sorted(list_of_dicts[0].keys())
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [float(d[k]) for d in list_of_dicts]
        m, s = _mean_std(vals)
        out[k] = {"mean": float(round(m, 6)), "std": float(round(s, 6))}
    return out


def _extract_result_macro(report: Dict[str, Any], kind: str) -> Dict[str, Dict[str, float]]:
    for r in report["results"]:
        if r["kind"] == kind:
            return r["macro_avg"]
    raise KeyError(f"Missing results kind='{kind}'")


def _extract_diag_macro(report: Dict[str, Any], split: str, phase: str) -> Dict[str, float]:
    block = report["diagnostics"][split][phase]
    macro = dict(block["macro_avg"])
    macro["GramCorr"] = float(block["gram_corr_mean"])
    macro["NeighOverlap"] = float(block["neighbor_overlap_mean"])
    macro["LangIDAcc"] = float(block["langid_probe"]["acc"])
    return macro


def _extract_pivot_macro(report: Dict[str, Any], split: str, phase: str) -> Dict[str, float]:
    return dict(report["pivot_retrieval"][split][phase]["macro_avg"])


def build_cv_report(
    fold_reports: List[Dict[str, Any]],
    cfg_common: Dict[str, Any],
    split: str = "holdout",
) -> Dict[str, Any]:
    # retrieval macro
    kind_t2i = f"{split}_t2i"
    kind_i2t = f"{split}_i2t"

    t2i_before = [_extract_result_macro(rep, kind_t2i)["before"] for rep in fold_reports]
    t2i_after = [_extract_result_macro(rep, kind_t2i)["after"] for rep in fold_reports]
    t2i_delta = [_extract_result_macro(rep, kind_t2i)["delta"] for rep in fold_reports]

    i2t_before = [_extract_result_macro(rep, kind_i2t)["before"] for rep in fold_reports]
    i2t_after = [_extract_result_macro(rep, kind_i2t)["after"] for rep in fold_reports]
    i2t_delta = [_extract_result_macro(rep, kind_i2t)["delta"] for rep in fold_reports]

    # diagnostics macro
    diag_before = [_extract_diag_macro(rep, split, "before") for rep in fold_reports]
    diag_after = [_extract_diag_macro(rep, split, "after") for rep in fold_reports]

    # pivot macro
    piv_before = [_extract_pivot_macro(rep, split, "before") for rep in fold_reports]
    piv_after = [_extract_pivot_macro(rep, split, "after") for rep in fold_reports]
    piv_delta = [{k: float(a[k] - b[k]) for k in a.keys()} for a, b in zip(piv_after, piv_before)]

    return {
        "config": {
            **cfg_common,
            "cv_n_folds": int(len(fold_reports)),
            "cv_split": split,
        },
        "macro_results": {
            "t2i": {
                "before": _agg_mean_std_dict(t2i_before),
                "after": _agg_mean_std_dict(t2i_after),
                "delta": _agg_mean_std_dict(t2i_delta),
            },
            "i2t": {
                "before": _agg_mean_std_dict(i2t_before),
                "after": _agg_mean_std_dict(i2t_after),
                "delta": _agg_mean_std_dict(i2t_delta),
            },
        },
        "macro_diagnostics": {
            "before": _agg_mean_std_dict(diag_before),
            "after": _agg_mean_std_dict(diag_after),
        },
        "macro_pivot": {
            "before": _agg_mean_std_dict(piv_before),
            "after": _agg_mean_std_dict(piv_after),
            "delta": _agg_mean_std_dict(piv_delta),
        },
    }


# ---------------------- Training utils ----------------------

def cosine_warmup(total_steps: int, base_lr: float, warmup: int):
    def get_lr(step: int) -> float:
        if step < warmup:
            return base_lr * (step + 1) / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return base_lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t)))
    return get_lr


def clip_loss(txt: torch.Tensor, img: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    txt = F.normalize(txt, dim=-1)
    img = F.normalize(img, dim=-1)
    logits = (txt @ img.T) / temp
    y = torch.arange(txt.size(0), device=txt.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


# ---------------------- Heads ----------------------

def make_eye_rect(d_text: int, d_img: int, device) -> torch.Tensor:
    eye_rect = torch.zeros(d_text, d_img, device=device)
    for i in range(min(d_text, d_img)):
        eye_rect[i, i] = 1.0
    return eye_rect


class LinearDeltaHead(nn.Module):
    def __init__(self, d_text: int, d_img: int, device):
        super().__init__()
        self.d_text = d_text
        self.d_img = d_img
        self.register_buffer("eye_rect", make_eye_rect(d_text, d_img, device=device))
        self.DeltaW = nn.Parameter(torch.zeros(d_text, d_img, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.eye_rect + self.DeltaW
        return x @ W

    def export_cpu(self) -> torch.Tensor:
        with torch.no_grad():
            return (self.eye_rect.detach().cpu() + self.DeltaW.detach().cpu())


class ResidualMLPHead(nn.Module):
    def __init__(self, d_text: int, d_img: int, hidden: int, dropout: float, device):
        super().__init__()
        self.d_text = d_text
        self.d_img = d_img
        self.hidden = hidden
        self.dropout = dropout
        self.register_buffer("eye_rect", make_eye_rect(d_text, d_img, device=device))

        self.fc1 = nn.Linear(d_text, hidden, bias=True, device=device)
        self.fc2 = nn.Linear(hidden, d_img, bias=True, device=device)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.eye_rect
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop(h)
        delta = self.fc2(h)
        return base + delta

    def export_payload(self) -> Dict[str, object]:
        return {
            "head": "mlp_residual",
            "d_text": int(self.d_text),
            "d_img": int(self.d_img),
            "hidden": int(self.hidden),
            "dropout": float(self.dropout),
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
        }


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
    ap.add_argument("--kfold", type=int, default=1)
    ap.add_argument("--fold_id", type=int, default=-1, help="If >=0, run only this fold (0..k-1)")
    ap.add_argument("--save_report", action="store_true")

    ap.add_argument("--prox_id", type=float, default=0.0)
    ap.add_argument("--ortho", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=0.0)
    ap.add_argument("--early_stop_metric", default="R@1", choices=["R@1", "MRR"])

    ap.add_argument("--diag_sample", type=int, default=2000)
    ap.add_argument("--diag_bins", type=int, default=30)
    ap.add_argument("--hub_k", type=int, default=10)
    ap.add_argument("--neigh_k", type=int, default=10)
    ap.add_argument("--no_langid_probe", action="store_true")
    ap.add_argument("--langid_sample_per_lang", type=int, default=2000)
    ap.add_argument("--langid_test_size", type=float, default=0.3)
    ap.add_argument("--langid_C", type=float, default=1.0)
    ap.add_argument("--langid_seed", type=int, default=0)

    ap.add_argument("--head", default="linear", choices=["linear", "mlp", "both"])
    ap.add_argument("--mlp_hidden", type=int, default=1024)
    ap.add_argument("--mlp_dropout", type=float, default=0.0)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    if len(langs) < 2:
        raise SystemExit("Need at least 2 languages")

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
        head_kind: str,
    ):
        N = I_shared.size(0)
        L = len(T_list)

        print(f"[{run_tag}][{head_kind}] start | N={N} | L={L} | d_text={d_text} | d_img={d_img} | device={args.device}")

        baseline_train_t2i = {langs[i]: retrieval_metrics(T_list[i], I_shared) for i in range(L)}
        baseline_train_i2t = {langs[i]: retrieval_metrics(I_shared, T_list[i]) for i in range(L)}

        diag_before = diagnostics_block(
            T_list,
            langs,
            sample=args.diag_sample,
            bins=args.diag_bins,
            hub_k=args.hub_k,
            neigh_k=args.neigh_k,
            do_langid_probe=(not args.no_langid_probe),
            langid_sample_per_lang=args.langid_sample_per_lang,
            langid_test_size=args.langid_test_size,
            langid_C=args.langid_C,
            langid_seed=args.langid_seed,
        )

        if head_kind == "linear":
            head = LinearDeltaHead(d_text, d_img, device=device).to(device)
            opt_params = [head.DeltaW]
            eye_square = torch.eye(d_img, device=device)
        elif head_kind == "mlp":
            head = ResidualMLPHead(
                d_text=d_text,
                d_img=d_img,
                hidden=args.mlp_hidden,
                dropout=args.mlp_dropout,
                device=device,
            ).to(device)
            opt_params = list(head.parameters())
            eye_square = None
        else:
            raise ValueError("head_kind must be 'linear' or 'mlp'")

        opt = torch.optim.Adam(opt_params, lr=args.lr, weight_decay=args.wd)

        total_steps = args.epochs * args.steps_per_epoch
        lr_fn = cosine_warmup(total_steps, args.lr, args.warmup)

        def set_lr(step: int):
            lr = lr_fn(step)
            for g in opt.param_groups:
                g["lr"] = lr

        I_dev = I_shared.to(device)
        T_dev = [t.to(device) for t in T_list]

        best_score = -1.0
        best_state: Optional[Dict[str, torch.Tensor]] = None

        def macro_score(metric_per_lang: Dict[str, Dict[str, float]], key: str) -> float:
            return float(np.mean([metric_per_lang[l][key] for l in metric_per_lang]))

        step = 0
        for epoch in range(args.epochs):
            loss_acc = 0.0
            loss_cnt = 0

            head.train()
            for _ in range(args.steps_per_epoch):
                idx = torch.randint(0, N, (args.batch_per_lang,), device=device)

                batch_t = torch.cat([T_dev[i][idx] for i in range(L)], dim=0)
                batch_i = I_dev[idx].repeat(L, 1)

                proj = head(batch_t)
                loss = clip_loss(proj, batch_i, temp=args.temp)

                if args.prox_id > 0.0:
                    if head_kind == "linear":
                        prox = torch.sum(head.DeltaW ** 2) / (d_text * d_img)
                    else:
                        s = 0.0
                        n = 0
                        for p in head.parameters():
                            s = s + torch.sum(p ** 2)
                            n += p.numel()
                        prox = s / max(1, n)
                    loss = loss + args.prox_id * prox

                if args.ortho > 0.0 and head_kind == "linear":
                    W = head.eye_rect + head.DeltaW
                    WT_W = W.t() @ W
                    ortho = torch.sum((WT_W - eye_square) ** 2) / (d_img * d_img)
                    loss = loss + args.ortho * ortho

                set_lr(step)
                opt.zero_grad()
                loss.backward()
                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(opt_params, args.max_grad_norm)
                opt.step()
                step += 1

                loss_acc += float(loss.detach().cpu().item())
                loss_cnt += 1

            print(f"[{run_tag}][{head_kind}] epoch={epoch} loss={loss_acc / max(1, loss_cnt):.4f}")

            if holdout_pair is not None:
                T_val, I_val = holdout_pair
                head.eval()
                with torch.no_grad():
                    T_proj_val = [F.normalize(head(t.to(device)), dim=-1).cpu() for t in T_val]
                val_metrics = {langs[i]: retrieval_metrics(T_proj_val[i], I_val) for i in range(L)}
                score = macro_score(val_metrics, args.early_stop_metric)
                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                    print(f"[{run_tag}][{head_kind}] earlystop update @ epoch={epoch} {args.early_stop_metric}={score:.4f}")

        if holdout_pair is not None and best_state is not None:
            head.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            print(f"[{run_tag}][{head_kind}] restored best state | best_{args.early_stop_metric}={best_score:.4f}")

        head.eval()
        with torch.no_grad():
            T_proj_train = [F.normalize(head(t.to(device)), dim=-1).cpu() for t in T_list]

        final_train_t2i = {langs[i]: retrieval_metrics(T_proj_train[i], I_shared) for i in range(L)}
        final_train_i2t = {langs[i]: retrieval_metrics(I_shared, T_proj_train[i]) for i in range(L)}

        diag_after = diagnostics_block(
            T_proj_train,
            langs,
            sample=args.diag_sample,
            bins=args.diag_bins,
            hub_k=args.hub_k,
            neigh_k=args.neigh_k,
            do_langid_probe=(not args.no_langid_probe),
            langid_sample_per_lang=args.langid_sample_per_lang,
            langid_test_size=args.langid_test_size,
            langid_C=args.langid_C,
            langid_seed=args.langid_seed,
        )

        report: Dict[str, Any] = {
            "config": {
                "out_dir": args.out_dir,
                "train_split": args.train_split,
                "langs": langs,
                "seed": args.seed,
                "device": args.device,
                "epochs": args.epochs,
                "steps_per_epoch": args.steps_per_epoch,
                "batch_per_lang": args.batch_per_lang,
                "lr": args.lr,
                "wd": args.wd,
                "temp": args.temp,
                "warmup": args.warmup,
                "prox_id": args.prox_id,
                "ortho": args.ortho if head_kind == "linear" else 0.0,
                "max_grad_norm": args.max_grad_norm,
                "early_stop_metric": args.early_stop_metric,
                "run_tag": run_tag,
                "head": head_kind,
                "mlp_hidden": args.mlp_hidden,
                "mlp_dropout": args.mlp_dropout,
            },
            "results": [],
            "diagnostics": {},
            "pivot_retrieval": {},
        }

        report["results"].append(_result_block("train", "train_t2i", I_shared.size(0), baseline_train_t2i, final_train_t2i))
        report["results"].append(_result_block("train", "train_i2t", I_shared.size(0), baseline_train_i2t, final_train_i2t))

        report["diagnostics"]["train"] = {"before": diag_before, "after": diag_after}
        report["pivot_retrieval"]["train"] = {
            "before": pivot_retrieval_block(T_list, I_shared, langs),
            "after": pivot_retrieval_block(T_proj_train, I_shared, langs),
        }

        if holdout_pair is not None:
            T_val, I_val = holdout_pair

            base_h_t2i = {langs[i]: retrieval_metrics(T_val[i], I_val) for i in range(L)}
            base_h_i2t = {langs[i]: retrieval_metrics(I_val, T_val[i]) for i in range(L)}

            with torch.no_grad():
                T_proj_h = [F.normalize(head(t.to(device)), dim=-1).cpu() for t in T_val]

            final_h_t2i = {langs[i]: retrieval_metrics(T_proj_h[i], I_val) for i in range(L)}
            final_h_i2t = {langs[i]: retrieval_metrics(I_val, T_proj_h[i]) for i in range(L)}

            report["results"].append(_result_block("holdout", "holdout_t2i", I_val.size(0), base_h_t2i, final_h_t2i))
            report["results"].append(_result_block("holdout", "holdout_i2t", I_val.size(0), base_h_i2t, final_h_i2t))

            diag_h_before = diagnostics_block(
                T_val,
                langs,
                sample=args.diag_sample,
                bins=args.diag_bins,
                hub_k=args.hub_k,
                neigh_k=args.neigh_k,
                do_langid_probe=(not args.no_langid_probe),
                langid_sample_per_lang=args.langid_sample_per_lang,
                langid_test_size=args.langid_test_size,
                langid_C=args.langid_C,
                langid_seed=args.langid_seed,
            )
            diag_h_after = diagnostics_block(
                T_proj_h,
                langs,
                sample=args.diag_sample,
                bins=args.diag_bins,
                hub_k=args.hub_k,
                neigh_k=args.neigh_k,
                do_langid_probe=(not args.no_langid_probe),
                langid_sample_per_lang=args.langid_sample_per_lang,
                langid_test_size=args.langid_test_size,
                langid_C=args.langid_C,
                langid_seed=args.langid_seed,
            )
            report["diagnostics"]["holdout"] = {"before": diag_h_before, "after": diag_h_after}

            report["pivot_retrieval"]["holdout"] = {
                "before": pivot_retrieval_block(T_val, I_val, langs),
                "after": pivot_retrieval_block(T_proj_h, I_val, langs),
            }

            macro_before = report["results"][-2]["macro_avg"]["before"]
            macro_after = report["results"][-2]["macro_avg"]["after"]
            print(f"[{run_tag}][{head_kind}] holdout_t2i macro before={macro_before} after={macro_after}")

        if head_kind == "linear":
            export = {"kind": "linear", "W": head.export_cpu(), "langs": langs, "train_split": args.train_split}
        else:
            export = {"kind": "mlp", **head.export_payload(), "langs": langs, "train_split": args.train_split}

        print(f"[{run_tag}][{head_kind}] done")
        return export, report

    def save_one(export: Dict[str, object], report: Dict[str, object], tag: str, head_kind: str):
        base = f"{args.train_split}_{'-'.join(langs)}_{tag}_{head_kind}"
        w_path = os.path.join(save_dir, f"W_{base}.pt")
        torch.save(export, w_path)
        report["config"]["weight_path"] = w_path
        rep_path = None
        if args.save_report:
            rep_path = os.path.join(save_dir, f"report_{base}.json")
            with open(rep_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        print(f"Saved W -> {w_path}")
        if rep_path is not None:
            print(f"Saved report -> {rep_path}")

    # -------- K-fold CV (internal eval only) --------
    if not eval_splits and args.kfold and args.kfold > 1:
        heads_to_run = ["linear", "mlp"] if args.head == "both" else [args.head]

        N_all = I_shared_all.size(0)
        if N_all < args.kfold:
            raise SystemExit("Not enough paired samples for requested k-fold")

        for head_kind in heads_to_run:
            kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

            fold_reports: List[Dict[str, Any]] = []
            cfg_common: Optional[Dict[str, Any]] = None

            for fold, (tr_idx, val_idx) in enumerate(kf.split(np.arange(N_all))):
                if args.fold_id >= 0 and fold != args.fold_id:
                    continue

                tr_idx = torch.as_tensor(tr_idx, dtype=torch.long)
                val_idx = torch.as_tensor(val_idx, dtype=torch.long)

                I_tr, I_val = I_shared_all[tr_idx], I_shared_all[val_idx]
                T_tr = [t[tr_idx] for t in T_list_all]
                T_val = [t[val_idx] for t in T_list_all]

                tag = f"fold{fold}"
                export, report = run_one_train(T_tr, I_tr, (T_val, I_val), run_tag=tag, head_kind=head_kind)
                save_one(export, report, tag=tag, head_kind=head_kind)

                if args.save_report:
                    fold_reports.append(report)
                    if cfg_common is None:
                        cfg_common = {
                            "out_dir": args.out_dir,
                            "train_split": args.train_split,
                            "langs": langs,
                            "seed": args.seed,
                            "device": args.device,
                            "head": head_kind,
                            "kfold": args.kfold,
                            "epochs": args.epochs,
                            "steps_per_epoch": args.steps_per_epoch,
                            "batch_per_lang": args.batch_per_lang,
                            "lr": args.lr,
                            "wd": args.wd,
                            "temp": args.temp,
                            "warmup": args.warmup,
                            "prox_id": args.prox_id,
                            "ortho": args.ortho if head_kind == "linear" else 0.0,
                            "max_grad_norm": args.max_grad_norm,
                            "early_stop_metric": args.early_stop_metric,
                            "mlp_hidden": args.mlp_hidden,
                            "mlp_dropout": args.mlp_dropout,
                            "diag_sample": args.diag_sample,
                            "diag_bins": args.diag_bins,
                            "hub_k": args.hub_k,
                            "neigh_k": args.neigh_k,
                            "langid_sample_per_lang": args.langid_sample_per_lang,
                            "langid_test_size": args.langid_test_size,
                            "langid_C": args.langid_C,
                            "langid_seed": args.langid_seed,
                        }

            # Write aggregated CV report (only if we actually ran multiple folds)
            if args.save_report and cfg_common is not None and len(fold_reports) > 0 and args.fold_id < 0:
                cv_rep = build_cv_report(fold_reports, cfg_common=cfg_common, split="holdout")
                cv_path = os.path.join(save_dir, f"report_cv_{args.train_split}_{'-'.join(langs)}_{head_kind}.json")
                with open(cv_path, "w", encoding="utf-8") as f:
                    json.dump(cv_rep, f, indent=2)
                print(f"Saved CV report -> {cv_path}")

        return

    # -------- Single run + optional external eval --------
    holdout_pair = None
    if not eval_splits:
        if not (0.0 < args.holdout < 1.0):
            raise SystemExit("--holdout must be in (0,1) when eval_splits is empty")
        N_all = I_shared_all.size(0)
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

    heads_to_run = ["linear", "mlp"] if args.head == "both" else [args.head]

    for head_kind in heads_to_run:
        export, report = run_one_train(T_list_run, I_run, holdout_pair, run_tag="single", head_kind=head_kind)
        save_one(export, report, tag="single", head_kind=head_kind)

        if eval_splits:
            for es in eval_splits:
                img_dir = os.path.join(args.out_dir, "embeddings_image", es)
                txt_dir = os.path.join(args.out_dir, "embeddings_text", es)
                if not (os.path.isdir(img_dir) and os.path.isdir(txt_dir)):
                    raise SystemExit(f"Missing embeddings dirs for eval split '{es}'")

                img_map_e, I_e = load_image_embeds(img_dir)
                T_e = load_text_embeds(txt_dir, langs)
                T_list_e, I_shared_e, _ = build_aligned_arrays(img_map_e, I_e, T_e, langs)

                if export["kind"] == "linear":
                    W = export["W"].float()
                    T_proj_e = [F.normalize(t @ W, dim=-1) for t in T_list_e]
                else:
                    head_cpu = ResidualMLPHead(
                        d_text=int(export["d_text"]),
                        d_img=int(export["d_img"]),
                        hidden=int(export["hidden"]),
                        dropout=float(export["dropout"]),
                        device=torch.device("cpu"),
                    )
                    head_cpu.load_state_dict(export["state_dict"])
                    head_cpu.eval()
                    with torch.no_grad():
                        T_proj_e = [F.normalize(head_cpu(t), dim=-1) for t in T_list_e]

                L = len(langs)
                base_e_t2i = {langs[i]: retrieval_metrics(T_list_e[i], I_shared_e) for i in range(L)}
                base_e_i2t = {langs[i]: retrieval_metrics(I_shared_e, T_list_e[i]) for i in range(L)}
                final_e_t2i = {langs[i]: retrieval_metrics(T_proj_e[i], I_shared_e) for i in range(L)}
                final_e_i2t = {langs[i]: retrieval_metrics(I_shared_e, T_proj_e[i]) for i in range(L)}

                report["results"].append(_result_block(es, f"eval_{es}_t2i", I_shared_e.size(0), base_e_t2i, final_e_t2i))
                report["results"].append(_result_block(es, f"eval_{es}_i2t", I_shared_e.size(0), base_e_i2t, final_e_i2t))

                diag_e_before = diagnostics_block(
                    T_list_e, langs,
                    sample=args.diag_sample, bins=args.diag_bins, hub_k=args.hub_k, neigh_k=args.neigh_k,
                    do_langid_probe=(not args.no_langid_probe),
                    langid_sample_per_lang=args.langid_sample_per_lang,
                    langid_test_size=args.langid_test_size,
                    langid_C=args.langid_C,
                    langid_seed=args.langid_seed,
                )
                diag_e_after = diagnostics_block(
                    T_proj_e, langs,
                    sample=args.diag_sample, bins=args.diag_bins, hub_k=args.hub_k, neigh_k=args.neigh_k,
                    do_langid_probe=(not args.no_langid_probe),
                    langid_sample_per_lang=args.langid_sample_per_lang,
                    langid_test_size=args.langid_test_size,
                    langid_C=args.langid_C,
                    langid_seed=args.langid_seed,
                )
                report["diagnostics"][f"eval:{es}"] = {"before": diag_e_before, "after": diag_e_after}

                report["pivot_retrieval"][f"eval:{es}"] = {
                    "before": pivot_retrieval_block(T_list_e, I_shared_e, langs),
                    "after": pivot_retrieval_block(T_proj_e, I_shared_e, langs),
                }

            base = f"{args.train_split}_{'-'.join(langs)}_single_{head_kind}"
            rep_path = os.path.join(save_dir, f"report_{base}.json")
            with open(rep_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"Saved report (with eval) -> {rep_path}")


if __name__ == "__main__":
    main()