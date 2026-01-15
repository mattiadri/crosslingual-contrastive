#!/usr/bin/env python3

import os
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import umap


# ---------------- I/O ----------------

def list_pt_files(dir_path: str) -> List[str]:
    return sorted(glob.glob(os.path.join(dir_path, "*.pt")))

def load_image_embeds(split_dir: str) -> Tuple[List[str], torch.Tensor]:
    pt_paths = list_pt_files(split_dir)
    if not pt_paths:
        raise SystemExit(f"No image embeddings found in {split_dir}")
    keys_all: List[str] = []
    embs_all: List[torch.Tensor] = []
    for p in pt_paths:
        payload = torch.load(p, map_location="cpu")
        E = payload["embeddings"].float()
        K = [str(k) for k in payload["keys"]]
        assert E.shape[0] == len(K)
        keys_all.extend(K)
        embs_all.append(E)
    I = torch.cat(embs_all, dim=0)
    I = F.normalize(I, dim=-1)
    return keys_all, I

def load_text_embeds(split_dir: str, langs: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
    pt_paths = list_pt_files(split_dir)
    if not pt_paths:
        raise SystemExit(f"No text embeddings found in {split_dir}")
    out = {l: {"keys": [], "emb": []} for l in langs}
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

def build_common(
    img_keys: List[str],
    I: torch.Tensor,
    T: Dict[str, Dict[str, torch.Tensor]],
    langs: List[str],
):
    img_map = {k: i for i, k in enumerate(img_keys)}
    sets = [set(T[l]["keys"]) for l in langs]
    common = set(img_map.keys()).intersection(*sets)
    if not common:
        raise SystemExit("No common keys across images and requested languages.")
    common = sorted(common)

    idx_img = torch.tensor([img_map[k] for k in common], dtype=torch.long)
    I_shared = I[idx_img]

    T_list = []
    for l in langs:
        first = {}
        for i, k in enumerate(T[l]["keys"]):
            if k not in first:
                first[k] = i
        rows = [first[k] for k in common]
        X = T[l]["emb"][rows]
        T_list.append(F.normalize(X, dim=-1))

    key_to_pos = {k: i for i, k in enumerate(common)}
    return common, key_to_pos, I_shared, T_list

# ---------------- Heads ----------------

class ResidualMLPHeadCPU(nn.Module):
    def __init__(self, d_text: int, d_img: int, hidden: int, dropout: float):
        super().__init__()
        eye_rect = torch.zeros(d_text, d_img)
        for i in range(min(d_text, d_img)):
            eye_rect[i, i] = 1.0
        self.register_buffer("eye_rect", eye_rect)
        self.fc1 = nn.Linear(d_text, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, d_img, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.eye_rect
        h = self.act(self.fc1(x))
        h = self.drop(h)
        return base + self.fc2(h)

def load_linear_W(path: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "W" in payload and torch.is_tensor(payload["W"]):
        return payload["W"].float()
    raise SystemExit(f"Bad linear weight file: {path}")

def load_mlp_head(path: str) -> ResidualMLPHeadCPU:
    payload = torch.load(path, map_location="cpu")
    if not (isinstance(payload, dict) and payload.get("kind") == "mlp"):
        raise SystemExit(f"Bad mlp weight file: {path}")
    head = ResidualMLPHeadCPU(
        d_text=int(payload["d_text"]),
        d_img=int(payload["d_img"]),
        hidden=int(payload["hidden"]),
        dropout=float(payload["dropout"]),
    )
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()
    return head

# ---------------- Utils ----------------

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def topk_idx(sim_row: np.ndarray, k: int) -> np.ndarray:
    k = min(k, sim_row.shape[0])
    return np.argpartition(-sim_row, kth=k - 1)[:k]

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="webdataset")
    ap.add_argument("--split", default="full")
    ap.add_argument("--langs", default="ar,de,en,es,fr,it,ja,pt,zh")
    ap.add_argument("--pivot", default="en", help="Language to visualize (only)")
    ap.add_argument("--linear_w", required=True)
    ap.add_argument("--mlp_w", required=True)
    ap.add_argument("--save_dir", default="paper/figs")

    ap.add_argument("--n_anchors", type=int, default=8)
    ap.add_argument("--k_img", type=int, default=140)
    ap.add_argument("--k_txt", type=int, default=140)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--umap_neighbors", type=int, default=25)
    ap.add_argument("--umap_min_dist", type=float, default=0.10)

    args = ap.parse_args()
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    if args.pivot not in langs:
        raise SystemExit(f"--pivot {args.pivot} not in langs {langs}")

    img_dir = os.path.join(args.out_dir, "embeddings_image", args.split)
    txt_dir = os.path.join(args.out_dir, "embeddings_text", args.split)
    if not os.path.isdir(img_dir):
        raise SystemExit(f"Missing: {img_dir}")
    if not os.path.isdir(txt_dir):
        raise SystemExit(f"Missing: {txt_dir}")
    if not os.path.isfile(args.linear_w):
        raise SystemExit(f"Missing linear_w: {args.linear_w}")
    if not os.path.isfile(args.mlp_w):
        raise SystemExit(f"Missing mlp_w: {args.mlp_w}")

    ensure_dir(args.save_dir)

    print("[load] embeddings...")
    img_keys, I = load_image_embeds(img_dir)
    T = load_text_embeds(txt_dir, langs)
    common, key_to_pos, I_shared, T_list = build_common(img_keys, I, T, langs)

    L = {l: i for i, l in enumerate(langs)}
    piv_i = L[args.pivot]
    T_piv = T_list[piv_i]
    N, D = T_piv.shape

    print("[load] heads...")
    W_lin = load_linear_W(args.linear_w)
    head_mlp = load_mlp_head(args.mlp_w)

    with torch.no_grad():
        T_lin = F.normalize(T_piv @ W_lin, dim=-1)
        T_mlp = F.normalize(head_mlp(T_piv), dim=-1)

    # Anchor selection: biggest gain to paired image
    sim_b = (T_piv * I_shared).sum(dim=1).cpu().numpy()
    sim_m = (T_mlp * I_shared).sum(dim=1).cpu().numpy()
    gain = sim_m - sim_b
    anchors = np.argsort(-gain)[: min(args.n_anchors, N)]
    print(f"[anchors] pivot={args.pivot} | selected {len(anchors)} by Δ(sim(text,image))")

    rng = np.random.default_rng(args.seed)

    ncols = 4
    nrows = int(np.ceil(len(anchors) / ncols))
    fig = plt.figure(figsize=(5.0 * ncols, 4.6 * nrows))

    for pi, a in enumerate(anchors):
        # Image neighborhood around paired image
        sim_img = (I_shared[a].unsqueeze(0) @ I_shared.T).squeeze(0).cpu().numpy()
        nn_img = topk_idx(sim_img, k=args.k_img)

        # Text subset aligned by index
        if len(nn_img) > args.k_txt:
            nn_txt = np.sort(rng.choice(nn_img, size=args.k_txt, replace=False))
        else:
            nn_txt = np.sort(nn_img)

        if a not in nn_txt:
            nn_txt[-1] = a
            nn_txt = np.sort(np.unique(nn_txt))

        I_loc = I_shared[nn_img]
        Tb_loc = T_piv[nn_txt]
        Tl_loc = T_lin[nn_txt]
        Tm_loc = T_mlp[nn_txt]

        X_fit = torch.cat([I_loc, Tb_loc, Tl_loc, Tm_loc], dim=0).cpu().numpy().astype(np.float32)
        X_fit = X_fit / (np.linalg.norm(X_fit, axis=1, keepdims=True) + 1e-12)

        reducer = umap.UMAP(
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric="cosine",
            random_state=args.seed,
            n_components=2,
            verbose=False,
        )
        Y = reducer.fit_transform(X_fit)

        nI = I_loc.size(0)
        nT = Tb_loc.size(0)
        Y_img = Y[:nI]
        Y_tb  = Y[nI:nI+nT]
        Y_lin = Y[nI+nT:nI+2*nT]
        Y_mlp = Y[nI+2*nT:nI+3*nT]

        a_pos = int(np.where(nn_txt == a)[0][0])

        ax = fig.add_subplot(nrows, ncols, pi + 1)

        c_before = "#E69F00"  # orange/yellow
        c_lin    = "#0072B2"  # blue
        c_mlp    = "#CC79A7"  # purple

        # Local text clouds (so you see the neighborhoods)
        ax.scatter(Y_img[:, 0], Y_img[:, 1], s=12, alpha=0.40, marker="x", color="k", label="images")
        ax.scatter(Y_tb[:, 0],  Y_tb[:, 1],  s=8, alpha=0.35, color=c_before, label=f"{args.pivot} before")
        ax.scatter(Y_lin[:, 0], Y_lin[:, 1], s=8, alpha=0.35, color=c_lin,    label=f"{args.pivot} after_linear")
        ax.scatter(Y_mlp[:, 0], Y_mlp[:, 1], s=8, alpha=0.35, color=c_mlp,    label=f"{args.pivot} after_mlp")


        # Paired image marker (exists because neighborhood built from it)
        ip = int(np.where(nn_img == a)[0][0]) if a in nn_img else None
        if ip is not None:
            ax.scatter([Y_img[ip, 0]], [Y_img[ip, 1]],
            s=140, alpha=0.95, marker="*",
            color="gold", edgecolors="k", linewidths=0.8,
            label="paired image")

        # Anchor markers
        ax.scatter([Y_tb[a_pos, 0]],  [Y_tb[a_pos, 1]],  s=40, alpha=0.95, marker="o",
           edgecolors="k", linewidths=0.6, label="anchor before", color=c_before)
        ax.scatter([Y_lin[a_pos, 0]], [Y_lin[a_pos, 1]], s=40, alpha=0.95, marker="s",
                edgecolors="k", linewidths=0.6, label="anchor after_linear", color=c_lin)
        ax.scatter([Y_mlp[a_pos, 0]], [Y_mlp[a_pos, 1]], s=40, alpha=0.95, marker="^",
                edgecolors="k", linewidths=0.6, label="anchor after_mlp", color=c_mlp)


        # Only anchor arrows
        arrow_scale = 10.0
        # before -> linear
        ax.arrow(
            Y_tb[a_pos, 0], Y_tb[a_pos, 1],
            arrow_scale * (Y_lin[a_pos, 0] - Y_tb[a_pos, 0]),
            arrow_scale * (Y_lin[a_pos, 1] - Y_tb[a_pos, 1]),
            length_includes_head=True,
            head_width=0.02, head_length=0.025,
            linewidth=0.5,
            alpha=0.6,
            color=c_lin
        )

        # before -> mlp
        ax.arrow(
            Y_tb[a_pos, 0], Y_tb[a_pos, 1],
            arrow_scale * (Y_mlp[a_pos, 0] - Y_tb[a_pos, 0]),
            arrow_scale * (Y_mlp[a_pos, 1] - Y_tb[a_pos, 1]),
            length_includes_head=True,
            head_width=0.02, head_length=0.025,
            linewidth=0.5,
            alpha=0.9,
            color=c_mlp
        )


        ax.set_title(f"anchor {pi+1} | Δsim={gain[a]:.4f}")
        ax.set_xticks([]); ax.set_yticks([])

        if pi == 0:
            ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    out_path = os.path.join(args.save_dir, f"umap_local_pivot_{args.pivot}_lin_vs_mlp.png")
    plt.savefig(out_path, dpi=500)
    plt.close()

    print("Saved:", out_path)

if __name__ == "__main__":
    main()

# python do_umap.py \
#   --out_dir webdataset --split full --langs ar,de,en,es,fr,it,ja,pt,zh --pivot en \
#   --linear_w webdataset/alignment/W_full_ar-de-en-es-fr-it-ja-pt-zh_fold0_linear.pt \
#   --mlp_w    webdataset/alignment/W_full_ar-de-en-es-fr-it-ja-pt-zh_fold0_mlp.pt \
#   --save_dir paper/figs