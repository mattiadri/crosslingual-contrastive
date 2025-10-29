import glob, os, io
from pathlib import Path
from PIL import Image
import webdataset as wds

def check_split(out_dir, split):
    urls = sorted(glob.glob(os.path.join(out_dir, split, "shard-*.tar")))
    print(f"\n[{split}] {len(urls)} shard")
    for tar in urls:
        bad = []
        for sample in wds.WebDataset(tar).to_tuple("__key__", "jpg;png;jpeg;webp"):
            try:
                k, img_bytes = sample
                Image.open(io.BytesIO(img_bytes)).verify()
            except Exception:
                bad.append(k)
        if bad:
            print(f"{Path(tar).name}: BAD={len(bad)} -> {bad[:10]}{' ...' if len(bad)>10 else ''}")
        else:
            print(f"{Path(tar).name}: OK")

if __name__ == "__main__":
    OUT_DIR = "webdataset"
    for split in ["train","val","test"]:
        check_split(OUT_DIR, split)