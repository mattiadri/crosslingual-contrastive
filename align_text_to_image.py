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
  --out_dir /home/mattia/crosslingual-contrastive/webdataset --train_split full --eval_splits "" \
  --langs ar,de,en,es,fr,it,ja,pt,zh --epochs 20 --steps_per_epoch 12 --batch_per_lang 32 \
  --lr 3e-3 --wd 1e-4 --temp 0.07 --warmup 60 --device cuda --save_report
"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

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


# ---------------------- Metrics ----------------------

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
        "kind": kind,              # "train" | "eval" | "holdout"
        "n_pairs": int(n_pairs),
        "before": before,
        "after": after,
        "delta": delta,
        "macro_avg": {
            "before": _macro_avg(before),
            "after":  _macro_avg(after),
            "delta":  _macro_avg(delta)
        }
    }
    if split is not None:          # include 'split' only when provided
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
    ap.add_argument("--eval_splits", default="", help="Comma-separated list; empty = use internal holdout")
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
    ap.add_argument("--save_report", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]

    # Train split: load and align by common keys
    train_img_dir = os.path.join(args.out_dir, "embeddings_image", args.train_split)
    train_txt_dir = os.path.join(args.out_dir, "embeddings_text", args.train_split)
    if not os.path.isdir(train_img_dir):
        raise SystemExit(f"Missing image embeddings dir: {train_img_dir}")
    if not os.path.isdir(train_txt_dir):
        raise SystemExit(f"Missing text embeddings dir: {train_txt_dir}")

    img_map, I = load_image_embeds(train_img_dir)
    T = load_text_embeds(train_txt_dir, langs)
    T_list, I_shared, _ = build_aligned_arrays(img_map, I, T, langs)

    # Evaluation setup: external splits or internal holdout
    eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
    holdout_pair = None
    if not eval_splits:
        if not (0.0 < args.holdout < 1.0):
            raise SystemExit("--holdout must be in (0,1) when eval_splits is empty")
        N_all = I_shared.size(0)
        if N_all < 2:
            raise SystemExit("Not enough paired samples after alignment")
        idx = np.arange(N_all)
        idx_tr, idx_val = train_test_split(idx, test_size=args.holdout, random_state=args.seed, shuffle=True)
        idx_tr  = torch.as_tensor(idx_tr, dtype=torch.long)
        idx_val = torch.as_tensor(idx_val, dtype=torch.long)
        I_tr, I_val = I_shared[idx_tr], I_shared[idx_val]
        T_tr = [t[idx_tr] for t in T_list]
        T_val = [t[idx_val] for t in T_list]
        T_list, I_shared = T_tr, I_tr
        holdout_pair = (T_val, I_val)

    N = I_shared.size(0)
    L = len(T_list)
    d_img = I_shared.size(1)
    d_text = T_list[0].size(1)
    device = torch.device(args.device)

    print(f"[train] aligned items: {N} | langs={langs} | d_text={d_text} | d_img={d_img}")

    baseline_train = {langs[i]: retrieval_metrics(T_list[i], I_shared) for i in range(L)}

    # Single linear projection W: text_d -> image_d
    W = nn.Parameter(torch.randn(d_text, d_img, device=device) / np.sqrt(d_text))
    opt = torch.optim.Adam([W], lr=args.lr, weight_decay=args.wd)

    total_steps = args.epochs * args.steps_per_epoch
    lr_fn = cosine_warmup(total_steps, args.lr, args.warmup)

    def set_lr(step):
        for g in opt.param_groups:
            g["lr"] = lr_fn(step)

    I_dev = I_shared.to(device)
    T_dev = [t.to(device) for t in T_list]

    step = 0
    for _ in range(args.epochs):
        for _ in range(args.steps_per_epoch):
            idx = torch.randint(0, N, (args.batch_per_lang,), device=device)
            batch_t = torch.cat([T_dev[i][idx] for i in range(L)], dim=0)
            batch_i = I_dev[idx].repeat(L, 1)
            loss = clip_loss(batch_t @ W, batch_i, temp=args.temp)
            set_lr(step)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1

    with torch.no_grad():
        T_proj_train = [F.normalize(t @ W, dim=-1).cpu() for t in T_dev]
    final_train = {langs[i]: retrieval_metrics(T_proj_train[i], I_shared) for i in range(L)}

    # Save W
    save_dir = os.path.join(args.out_dir, "alignment")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    w_path = os.path.join(save_dir, f"W_{args.train_split}_{'-'.join(langs)}.pt")
    torch.save({"W": W.detach().cpu(), "langs": langs, "train_split": args.train_split}, w_path)
    print(f"Saved W -> {w_path}")

    # Report (v2, compact)
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
        "weight_path": w_path
    }
    if not eval_splits:
        config["holdout"] = args.holdout
    else:
        config["eval_splits"] = eval_splits

    report = {"version": "v2", "config": config, "results": []}

    # Train block: no 'split' key
    report["results"].append(
        _result_block(None, "train", I_shared.size(0), baseline_train, final_train)
    )

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
            T_proj_e = [F.normalize(t @ W.detach().cpu(), dim=-1) for t in T_list_e]
        final_e = {langs[i]: retrieval_metrics(T_proj_e[i], I_shared_e) for i in range(len(langs))}
        report["results"].append(
            _result_block(es, "eval", I_shared_e.size(0), base_e, final_e)
        )
        print(f"[eval:{es}] done. common={len(common_e)}")

    # Internal holdout
    if holdout_pair is not None:
        T_val, I_val = holdout_pair
        base_h = {langs[i]: retrieval_metrics(T_val[i], I_val) for i in range(len(langs))}
        with torch.no_grad():
            T_proj_h = [F.normalize(t @ W.detach().cpu(), dim=-1) for t in T_val]
        final_h = {langs[i]: retrieval_metrics(T_proj_h[i], I_val) for i in range(len(langs))}
        report["results"].append(
            _result_block("holdout", "holdout", I_val.size(0), base_h, final_h)
        )
        print(f"[eval:holdout] done. common={I_val.size(0)}")

    if args.save_report:
        rep_path = os.path.join(save_dir, f"report_{args.train_split}_{'-'.join(langs)}.json")
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report -> {rep_path}")


if __name__ == "__main__":
    main()
