#!/usr/bin/env python3
"""
Embed multilingual captions from existing WebDataset shards into the same CLIP space
as your image embeddings. Uses a CLIP-compatible multilingual text encoder
(default: sentence-transformers/clip-ViT-B-32-multilingual-v1).

For each shard, saves a .pt file at:
  OUT_DIR/embeddings_text/{split}/{shard}.pt

Saved payload per shard:
{
  "text_model": <model_id>,
  "split": <split>,
  "shard": <shard>,
  "langs": [<langs>],
  "keys": {<lang>: [k1, k2, ...]},
  "embeddings": {<lang>: torch.FloatTensor [N_lang, D]},
  "dim": D,
  "note": "unit-norm; CLIP-compatible; cosine-ready"
}

Example:
python emb_from_text.py \
  --out_dir /home/mattia/crosslingual-contrastive/webdataset \
  --splits train,val,test \
  --text_model sentence-transformers/clip-ViT-B-32-multilingual-v1 \
  --langs en,fr,es,pt,zh,ar,ja,it,de \
  --batch 1024 --device cuda
"""

import os
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import webdataset as wds
from sentence_transformers import SentenceTransformer

# ---------------- Discovery & FS ----------------

def list_shards(out_dir: str, split: str) -> List[str]:
    return sorted(glob.glob(os.path.join(out_dir, split, "shard-*.tar")))


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


# ---------------- Caption extraction ----------------

PREFERRED_FIELDS = ["cap_best", "cap_ref", "cap_attr", "cap_alt", "title"]


def pick_caption(sample: Dict[str, bytes], lang: str) -> Optional[str]:
    for base in PREFERRED_FIELDS:
        k = f"{base}_{lang}.txt"
        if k in sample:
            try:
                txt = sample[k].decode("utf-8", errors="ignore").strip()
            except Exception:
                txt = None
            if txt:
                return txt
    return None


def load_captions_from_shard(shard_path: str, langs: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Return (keys_by_lang, texts_by_lang). Lists are per-language aligned."""
    keys_by_lang = {l: [] for l in langs}
    texts_by_lang = {l: [] for l in langs}

    for sample in wds.WebDataset(shard_path, shardshuffle=False, resampled=False, empty_check=False):
        key = sample.get("__key__")
        if key is None:
            continue
        if isinstance(key, bytes):
            key = key.decode("utf-8", errors="ignore")
        for l in langs:
            cap = pick_caption(sample, l)
            if cap:
                keys_by_lang[l].append(key)
                texts_by_lang[l].append(cap)
    return keys_by_lang, texts_by_lang


# ---------------- Embedding ----------------

@torch.inference_mode()
def embed_texts(texts: List[str], model: SentenceTransformer, device: str, batch: int, normalize: bool = True) -> torch.Tensor:
    if not texts:
        dim = model.get_sentence_embedding_dimension()
        return torch.empty((0, dim), dtype=torch.float32)
    emb = model.encode(
        texts,
        batch_size=batch,
        convert_to_tensor=True,
        device=device,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    emb = emb.to(torch.float32)
    if normalize:
        emb = F.normalize(emb, dim=-1)
    return emb


def process_split(split: str, args, txt_model: SentenceTransformer) -> None:
    urls = list_shards(args.out_dir, split)
    if not urls:
        print(f"[info] No shards found in {split}/ — skipping.")
        return

    save_root = os.path.join(args.out_dir, "embeddings_text", split)
    ensure_dir(save_root)
    print(f"\n=== Split: {split} | {len(urls)} shard(s) ===")

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    print(f"Languages: {', '.join(langs)}")

    for shard_path in urls:
        shard_name = Path(shard_path).stem.replace(".tar", "")
        save_path = os.path.join(save_root, f"{shard_name}.pt")
        if (not args.overwrite) and os.path.isfile(save_path):
            print(f"[skip] {split}/{shard_name} already exists: {save_path}")
            continue

        keys_by_lang, texts_by_lang = load_captions_from_shard(shard_path, langs)

        emb_by_lang: Dict[str, torch.Tensor] = {}
        dim: Optional[int] = None
        total_caps = 0
        for l in langs:
            texts = texts_by_lang[l]
            if texts:
                embs = embed_texts(texts, txt_model, args.device, args.batch, normalize=True)
                emb_by_lang[l] = embs.cpu()
                if dim is None:
                    dim = int(embs.shape[1])
                total_caps += len(texts)
            else:
                emb_by_lang[l] = torch.empty((0, txt_model.get_sentence_embedding_dimension()))
                if dim is None:
                    dim = txt_model.get_sentence_embedding_dimension()

        payload = {
            "text_model": args.text_model,
            "split": split,
            "shard": shard_name,
            "langs": langs,
            "keys": keys_by_lang,
            "embeddings": emb_by_lang,
            "dim": dim,
            "note": "unit-norm features; cosine-ready; CLIP-compatible",
        }

        tmp_path = save_path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, save_path)
        print(f"[ok] {split}/{shard_name}: {total_caps} captions embedded (D={dim}) -> {save_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Root folder containing {split}/shard-*.tar.")
    ap.add_argument("--splits", default="all", help='Comma-separated (e.g., "train,val,test") or "all".')
    ap.add_argument("--text_model", default="sentence-transformers/clip-ViT-B-32-multilingual-v1",
                    help="Sentence-Transformers model id compatible with CLIP image encoder.")
    ap.add_argument("--langs", default="en,fr,es,pt,zh,ar,ja,it,de",
                    help="Comma-separated ISO-639-1 language codes to extract (suffix in *_<lang>.txt)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.out_dir):
        raise SystemExit(f"OUT_DIR does not exist: {args.out_dir}")

    splits = ["train", "val", "test"] if args.splits.lower() == "all" else [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise SystemExit("No split specified.")

    print(f"Loading text model: {args.text_model} on {args.device} ...")
    txt_model = SentenceTransformer(args.text_model, device=args.device)
    txt_model.eval()

    print(f"OUT_DIR: {args.out_dir}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Encoder: {args.text_model} | Device: {args.device}")

    for split in splits:
        process_split(split, args, txt_model)

    print("\nAll done.")


if __name__ == "__main__":
    main()
