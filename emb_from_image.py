#!/usr/bin/env python3
"""
Embed images from WebDataset shards using OpenCLIP (image -> CLIP space),
normalized (unit-norm) for cosine similarity.

Output: .pt files per shard at OUT_DIR/embeddings_image/{split}/{shard}.pt
Contains: {model_name, pretrained, split, shard, keys, embeddings [N,D], dim, note}

Example:
python emb_from_image.py \
  --out_dir webdataset/ \
  --splits full \
  --clip_model ViT-B-32 \
  --clip_pretrained openai \
  --batch 256 --num_workers 4 --device cuda

For multilingual text queries, use a compatible text encoder (e.g.
  sentence-transformers/clip-ViT-B-32-multilingual-v1) to map text to the same space.
"""

import os
import io
import glob
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import webdataset as wds
from PIL import Image
from torchvision import transforms
import open_clip


# ---------------- IO & small utils ----------------
def pil_decode(img_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def list_shards(out_dir: str, split: str) -> List[str]:
    return sorted(glob.glob(os.path.join(out_dir, split, "shard-*.tar")))


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


# ---------------- Dataset builder ----------------
def build_wds(url: str, img_tf: transforms.Compose, exts: str = "jpg;jpeg;png"):
    """Create a WebDataset that yields (__key__, image_tensor)."""
    return (
        wds.WebDataset(
            url,
            shardshuffle=False,
            resampled=False,
            empty_check=False
        )
        .to_tuple("__key__", exts)
        .map_tuple(lambda k: k, lambda img: img_tf(pil_decode(img)))
    )


# ---------------- Embedding ----------------
@torch.inference_mode()
def embed_batch(imgs: torch.Tensor, model, device: str) -> torch.Tensor:
    feats = model.encode_image(imgs.to(device))  # [B, D]
    return F.normalize(feats, dim=-1)


def process_split(split: str, args, img_encoder, img_tf, device: str) -> None:
    urls = list_shards(args.out_dir, split)
    if not urls:
        print(f"[info] No shards found in {split}/ — skipping.")
        return

    save_root = os.path.join(args.out_dir, "embeddings_image", split)
    ensure_dir(save_root)
    print(f"=== Split: {split} | {len(urls)} shard(s) ===")

    is_cuda = device.startswith("cuda")
    amp_dtype = torch.float16 if is_cuda else torch.bfloat16

    for shard_path in urls:
        shard_name = Path(shard_path).stem.replace(".tar", "")
        save_path = os.path.join(save_root, f"{shard_name}.pt")
        if (not args.overwrite) and os.path.isfile(save_path):
            print(f"[skip] {split}/{shard_name} already exists: {save_path}")
            continue

        ds = build_wds(shard_path, img_tf)

        # With a single shard per loop iteration, keep workers at 0/1 to avoid empty partitions
        workers = min(args.num_workers, 1)
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch,
            num_workers=workers,
            pin_memory=is_cuda,
            shuffle=False,
            drop_last=False,
        )

        all_keys, all_embs = [], []
        n_items = 0

        # Autocast on CUDA only
        if is_cuda:
            cm = torch.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            from contextlib import nullcontext
            cm = nullcontext()

        with cm:
            for keys, imgs in loader:
                embs = embed_batch(imgs, img_encoder, device)
                all_keys.extend(list(keys))
                all_embs.append(embs.cpu())
                n_items += len(keys)

        if n_items == 0:
            print(f"[warn] {split}/{shard_name} empty? No valid images.")
            continue

        embs = torch.cat(all_embs, dim=0)
        payload = {
            "model_name": args.clip_model,
            "pretrained": args.clip_pretrained,
            "split": split,
            "shard": shard_name,
            "keys": all_keys,
            "embeddings": embs,
            "dim": int(embs.shape[1]),
            "note": "unit-norm features (CLIP image encoder); cosine similarity ready",
        }

        # Atomic save
        tmp_path = save_path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, save_path)
        print(f"[ok] {split}/{shard_name}: {embs.shape[0]} embeddings (D={embs.shape[1]}) -> {save_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Root folder of the shards (your build script's OUT_DIR).")
    ap.add_argument("--splits", default="full", help='Comma-separated (e.g., "train,val,test") or "all".')
    ap.add_argument("--clip_model", default="ViT-B-32", help="OpenCLIP model name (e.g., ViT-B-32, ViT-B-16, ViT-L-14)")
    ap.add_argument("--clip_pretrained", default="openai", help="Pretrained tag (e.g., openai, laion2b_s34b_b79k, ...)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.out_dir):
        raise SystemExit(f"OUT_DIR does not exist: {args.out_dir}")

    splits = ["train", "val", "test"] if args.splits.lower() == "all" else [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise SystemExit("No split specified.")

    print(f"Loading OpenCLIP: {args.clip_model} ({args.clip_pretrained}) on {args.device} ...")
    img_encoder, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, device=args.device
    )
    img_encoder.eval()
    for p in img_encoder.parameters():
        p.requires_grad = False

    img_tf = preprocess

    print(f"OUT_DIR: {args.out_dir}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Encoder: OpenCLIP {args.clip_model} | Pretrained: {args.clip_pretrained} | Device: {args.device}")

    for split in splits:
        process_split(split, args, img_encoder, img_tf, args.device)

    print("All done.")


if __name__ == "__main__":
    main()