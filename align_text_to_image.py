#!/usr/bin/env python3
"""
Align multilingual caption embeddings to the image embedding space using images as the pivot.

Training happens on one split (e.g., train).

Artifacts written under --out_dir:
- alignment/W_<train_split>_<langs>.pt
- alignment/report_<train_split>_<langs>.json  (if --save_report)

Requires the precomputed embeddings directories produced earlier:
- embeddings_image/<split>/*.pt      (image vectors)
- embeddings_text/<split>/*.pt       (text vectors per language)

python align_text_to_image.py \
  --out_dir /home/mattia/crosslingual-contrastive/webdataset \
  --train_split train \
  --eval_splits val,test \
  --langs ar,de,en,es,fr,it,ja,pt,zh \
  --epochs 20 --steps_per_epoch 12 --batch_per_lang 32 \
  --lr 3e-3 --wd 1e-4 --temp 0.07 --warmup 60 \
  --device cuda \
  --save_report
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
        if out[l]["emb"]:
            out[l]["emb"] = torch.cat(out[l]["emb"], dim=0)
        else:
            out[l]["emb"] = torch.empty(0, 1)
    return out


def build_aligned_arrays(img_map: Dict[str, int], I: torch.Tensor, T: Dict[str, Dict[str, torch.Tensor]], langs: List[str]):
    sets = [set(T[l]["keys"]) for l in langs]
    common = set(img_map.keys()).intersection(*sets)
    if not common:
        raise SystemExit("No common keys across images and all requested languages.")
    common = sorted(common)
    idx_img = torch.tensor([img_map[k] for k in common], dtype=torch.long)
    I_shared = I[idx_img]
    T_list = []
    for l in langs:
        seen = {}
        for i, k in enumerate(T[l]["keys"]):
            if k not in seen:
                seen[k] = i
        rows = [seen[k] for k in common]
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


def lang_probe_acc(X_list: List[torch.Tensor], steps=200, lr=0.08):
    X = torch.cat([x.detach() for x in X_list], dim=0)
    y = torch.cat([torch.full((x.size(0),), i, dtype=torch.long) for i, x in enumerate(X_list)], dim=0)
    Wp = torch.zeros(X.size(1), len(X_list), requires_grad=True)
    opt = torch.optim.SGD([Wp], lr=lr)
    for _ in range(steps):
        logits = X @ Wp
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = (X @ Wp).argmax(1)
        return (pred == y).float().mean().item()


def crosslingual_image_agreement(text_list: List[torch.Tensor], image: torch.Tensor):
    idxs = []
    for t in text_list:
        sims = t @ image.T
        idxs.append(torch.argmax(sims, dim=1))
    agree = (idxs[0] == idxs[1])
    for k in range(2, len(idxs)):
        agree = agree & (idxs[k] == idxs[0])
    return agree.float().mean().item()


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
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--eval_splits", default="val,test", help="Comma-separated list; empty to skip eval")
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
    ap.add_argument("--save_report", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]

    # ---------- Train on train_split ----------
    train_img_dir = os.path.join(args.out_dir, "embeddings_image", args.train_split)
    train_txt_dir = os.path.join(args.out_dir, "embeddings_text", args.train_split)
    if not os.path.isdir(train_img_dir):
        raise SystemExit(f"Missing image embeddings dir: {train_img_dir}")
    if not os.path.isdir(train_txt_dir):
        raise SystemExit(f"Missing text embeddings dir: {train_txt_dir}")

    img_map, I = load_image_embeds(train_img_dir)
    T = load_text_embeds(train_txt_dir, langs)
    T_list, I_shared, common_keys = build_aligned_arrays(img_map, I, T, langs)

    N = I_shared.size(0)
    L = len(T_list)
    d_img = I_shared.size(1)
    d_text = T_list[0].size(1)
    device = torch.device(args.device)

    print(f"[train] aligned items: {N} | langs={langs} | d_text={d_text} | d_img={d_img}")

    # Baselines (train)
    baseline_train = {langs[i]: retrieval_metrics(T_list[i], I_shared) for i in range(L)}

    # Model W (put on the chosen device)
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

    report = {
        "train_split": args.train_split,
        "langs": langs,
        "train_before": baseline_train,
        "train_after": final_train,
    }

    # ---------- Evaluate on eval_splits ----------
    eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
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
        # Baseline on eval
        base_e = {langs[i]: retrieval_metrics(T_list_e[i], I_shared_e) for i in range(L)}
        # Project with learned W
        with torch.no_grad():
            T_proj_e = [F.normalize(t @ W.detach().cpu(), dim=-1) for t in T_list_e]
        final_e = {langs[i]: retrieval_metrics(T_proj_e[i], I_shared_e) for i in range(L)}
        report[f"eval_{es}_before"] = base_e
        report[f"eval_{es}_after"] = final_e
        print(f"[eval:{es}] done. common={len(common_e)}")

    if args.save_report:
        rep_path = os.path.join(save_dir, f"report_{args.train_split}_{'-'.join(langs)}.json")
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report -> {rep_path}")


if __name__ == "__main__":
    main()