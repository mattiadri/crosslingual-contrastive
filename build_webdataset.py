#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import io
import glob
import json
import hashlib
import random
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from urllib.parse import urlparse, quote, unquote

import polars as pl
from tqdm import tqdm
import aiohttp
import uvloop
import orjson as _json
import webdataset as wds
from PIL import Image, ImageOps
import cairosvg

# ----------------------------
# Configuration
# ----------------------------

DATA_DIR   = "data"           # folder containing WIT TSVs
OUT_DIR    = "webdataset"     # output folder for webdataset .tar shards
STATE_DIR  = "state"          # progress/resume + failure logs

# Required languages (ISO-639-1)
REQUIRED_LANGS = ["en", "fr", "es", "pt", "zh", "ar", "ja", "it", "de"]

# WIT columns
COL_LANGUAGE   = "language"
COL_IMAGE_URL  = "image_url"
COL_PAGE_TITLE = "page_title"
COL_CAP_ALT    = "caption_alt_text_description"
COL_CAP_REF    = "caption_reference_description"
COL_CAP_ATTR   = "caption_attribution_description"

# Preference order for the "best" caption
PREFERRED_ORDER = ["cap_ref", "cap_attr", "cap_alt"]

# Language-level validity policy
VALIDITY_MODE = "single"               # "any" | "single"
REQUIRED_FIELD_FOR_SINGLE = "cap_ref"  # used only if VALIDITY_MODE == "single"

# Sharding & download
SHARD_SIZE = 2000
MAX_CONC_DOWNLOADS = 12          # global concurrency
PER_HOST_LIMIT = 4               # per-domain cap (prevents 403/429)
DOWNLOAD_TIMEOUT = 40            # seconds
RETRIES = 4
TTL_DNS_CACHE = 300              # DNS cache TTL (seconds)

# Image robustness
MAX_IMAGE_BYTES = 65_000_000     # raise if needed
REENCODE_TO = "jpeg"             # "jpeg" | "png" | "keep"
JPEG_QUALITY = 95

# Input patterns per split
SPLIT_PATTERNS = {
    "train": "wit_v1.train.all-*.tsv",
    "val":   "wit_v1.val.all-*.tsv",
    "test":  "wit_v1.test.all-*.tsv",
}

# Retry policy
RETRIABLE_STATUSES = {429, 500, 502, 503, 504}
NONRETRIABLE_STATUSES = {400, 401, 403, 404, 410}
HOST_HARDFAIL_THRESHOLD = 20   # after N hard fails, skip the domain for this run

UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]

# Wikimedia helpers
WIKIMEDIA_NETLOC = {"upload.wikimedia.org"}
MEDIAWIKI_API = "https://commons.wikimedia.org/w/api.php"

# ----------------------------
# Utilities
# ----------------------------

def dumps(obj):  # orjson, compatible with ShardWriter
    return _json.dumps(obj)

def ensure_dirs():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(STATE_DIR).mkdir(parents=True, exist_ok=True)

def stable_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]

def detect_next_shard_index(out_split_dir: Path) -> int:
    out_split_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_split_dir.glob("shard-*.tar"))
    if not existing:
        return 0
    m = re.search(r"shard-(\d+)\.tar$", existing[-1].name)
    return int(m.group(1)) + 1 if m else 0

def load_done_set(split: str) -> Set[str]:
    p = Path(STATE_DIR) / f"{split}_done.txt"
    if not p.exists(): return set()
    with p.open("r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def append_done(split: str, url: str):
    p = Path(STATE_DIR) / f"{split}_done.txt"
    with p.open("a", encoding="utf-8") as f:
        f.write(url + "\n")

def list_tsvs_for_split(split: str) -> List[str]:
    return sorted(glob.glob(str(Path(DATA_DIR) / SPLIT_PATTERNS[split])))

def language_to_suffix(lang: str) -> str:
    return lang

def looks_like_html(b: bytes) -> bool:
    head = b[:512].lower()
    return b"<html" in head or b"<!doctype html" in head

def _origin(url: str) -> str:
    u = urlparse(url)
    return f"{u.scheme}://{u.netloc}"

def load_retry_set(split: str) -> Set[str]:
    p = Path(STATE_DIR) / f"{split}_retry.txt"
    if not p.exists():
        return set()
    with p.open("r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def normalize_url(u: str) -> str:
    """upgrade http→https for Wikimedia to avoid 404/mixed content"""
    p = urlparse(u)
    if p.netloc in WIKIMEDIA_NETLOC and p.scheme == "http":
        return u.replace("http://", "https://", 1)
    return u

def _wikimedia_filename_from_url(u: str) -> Optional[str]:
    try:
        path = urlparse(u).path
        name = path.split("/")[-1]
        return name or None
    except Exception:
        return None

def commons_title_from_url(u: str) -> Optional[str]:
    """Extract 'File:<name>' title for Commons API (handles percent-encoding)."""
    try:
        path = urlparse(u).path
        name = path.split("/")[-1]
        if not name:
            return None
        name = unquote(name).replace(" ", "_")
        return f"File:{name}"
    except Exception:
        return None

# ----------------------------
# Polars pipeline
# ----------------------------

def _nonempty(expr: pl.Expr) -> pl.Expr:
    return pl.when(expr.is_not_null() & (expr.str.len_bytes() > 0)).then(expr).otherwise(None)

def read_valid_multilingual_samples(tsv_files: List[str]) -> pl.DataFrame:
    """
    Returns DF with:
      - image_url
      - langpack_structs: list of struct {lang, title, cap_alt, cap_ref, cap_attr, cap_best}
    Validity:
      - any: cap_best must be present
      - single: REQUIRED_FIELD_FOR_SINGLE must be present
    Then filters to URLs that cover all REQUIRED_LANGS.
    """
    if not tsv_files:
        return pl.DataFrame({COL_IMAGE_URL: [], "langpack_structs": []})

    order_cols = {"cap_alt": "cap_alt", "cap_ref": "cap_ref", "cap_attr": "cap_attr"}
    single_field_col = order_cols.get(REQUIRED_FIELD_FOR_SINGLE, "cap_ref")

    lazies = []
    for tsv in tsv_files:
        lf = (
            pl.scan_csv(
                tsv,
                separator="\t",
                has_header=True,
                infer_schema_length=0,
                ignore_errors=True,
                try_parse_dates=False,
            )
            .select([
                pl.col(COL_LANGUAGE).alias("lang"),
                pl.col(COL_IMAGE_URL),
                pl.col(COL_PAGE_TITLE).alias("title"),
                pl.col(COL_CAP_ALT).alias("cap_alt"),
                pl.col(COL_CAP_REF).alias("cap_ref"),
                pl.col(COL_CAP_ATTR).alias("cap_attr"),
            ])
            .with_columns([
                pl.col("title").fill_null("").str.strip_chars().alias("title"),
                pl.col("cap_alt").fill_null("").str.strip_chars().alias("cap_alt"),
                pl.col("cap_ref").fill_null("").str.strip_chars().alias("cap_ref"),
                pl.col("cap_attr").fill_null("").str.strip_chars().alias("cap_attr"),
            ])
        )

        best_exprs = [_nonempty(pl.col(order_cols[name])) for name in PREFERRED_ORDER]
        lf = lf.with_columns([
            pl.coalesce(best_exprs).alias("cap_best")
        ])

        if VALIDITY_MODE == "any":
            validity_filter = (
                pl.col("lang").is_in(REQUIRED_LANGS)
                & pl.col(COL_IMAGE_URL).is_not_null()
                & (pl.col(COL_IMAGE_URL).str.len_bytes() > 0)
                & pl.col("cap_best").is_not_null()
            )
        else:  # "single"
            validity_filter = (
                pl.col("lang").is_in(REQUIRED_LANGS)
                & pl.col(COL_IMAGE_URL).is_not_null()
                & (pl.col(COL_IMAGE_URL).str.len_bytes() > 0)
                & _nonempty(pl.col(single_field_col)).is_not_null()
            )

        lf = lf.filter(validity_filter).select([
            "lang", COL_IMAGE_URL, "title", "cap_alt", "cap_ref", "cap_attr", "cap_best"
        ])
        lazies.append(lf)

    lf_all = pl.concat(lazies, how="diagonal")

    lf_grouped = (
        lf_all
        .group_by(COL_IMAGE_URL, maintain_order=False)
        .agg([
            pl.col("lang").unique().alias("langs_ok"),
            pl.struct(["lang", "title", "cap_alt", "cap_ref", "cap_attr", "cap_best"]).alias("langpack_structs"),
        ])
        .with_columns([
            pl.col("langs_ok").list.len().alias("nlangs"),
        ])
        .filter(pl.col("nlangs") >= len(REQUIRED_LANGS))
        .filter(pl.col("langs_ok").list.set_intersection(pl.lit(REQUIRED_LANGS)).list.len() == len(REQUIRED_LANGS))
    )

    df = lf_grouped.collect()
    return df.select([COL_IMAGE_URL, "langpack_structs"])


def structs_to_langdict(structs: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for s in structs:
        lang = s.get("lang")
        if lang in REQUIRED_LANGS:
            out[lang] = {
                "title":    s.get("title") or "",
                "cap_alt":  s.get("cap_alt") or "",
                "cap_ref":  s.get("cap_ref") or "",
                "cap_attr": s.get("cap_attr") or "",
                "cap_best": s.get("cap_best") or "",
            }
    return out

# ----------------------------
# Shard writing
# ----------------------------

class ShardManager:
    def __init__(self, out_dir: Path, start_index: int, shard_size: int):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(self.out_dir / "shard-%05d.tar")
        self.writer = wds.ShardWriter(
            pattern,
            maxcount=shard_size,
            start_shard=start_index,
        )

    def write_sample(self, key: str, image_bytes: bytes, meta: Dict, img_ext: str = ".jpg"):
        ext_key = img_ext.lstrip(".").lower()
        sample = {
            "__key__": key,
            ext_key: image_bytes,
            "json": dumps({"url": meta["url"], "languages": meta["languages"]}),
        }
        for lang, pack in meta["by_lang"].items():
            suf = language_to_suffix(lang)
            if pack.get("cap_best"):
                sample[f"cap_best_{suf}.txt"] = pack["cap_best"].encode("utf-8")
            if pack.get("title"):
                sample[f"title_{suf}.txt"] = pack["title"].encode("utf-8")
            if pack.get("cap_attr"):
                sample[f"cap_attr_{suf}.txt"] = pack["cap_attr"].encode("utf-8")
            if pack.get("cap_ref"):
                sample[f"cap_ref_{suf}.txt"] = pack["cap_ref"].encode("utf-8")
            if pack.get("cap_alt"):
                sample[f"cap_alt_{suf}.txt"] = pack["cap_alt"].encode("utf-8")

        self.writer.write(sample)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

# ----------------------------
# Image helpers (SVG/TIFF/validation)
# ----------------------------

def is_svg_bytes(b: bytes) -> bool:
    head = b[:200].lstrip().lower()
    return head.startswith(b"<?xml") or b"<svg" in head

def rasterize_svg_to_png(b: bytes) -> bytes:
    # Render to PNG; PNG will then be re-encoded according to REENCODE_TO
    return cairosvg.svg2png(bytestring=b, dpi=96)

def is_tiff_bytes(b: bytes) -> bool:
    # TIFF magics: II*\x00 or MM\x00*
    return b[:4] in (b"II*\x00", b"MM\x00*")

def is_valid_image_bytes(b: bytes) -> bool:
    """
    Validate bytes as an image, filter HTML/JSON, handle alpha and EXIF safely.
    """
    if b is None or len(b) < 64:
        return False
    if looks_like_html(b):
        return False
    try:
        im = Image.open(io.BytesIO(b))
        im.verify()  # integrity check
        im2 = Image.open(io.BytesIO(b))
        im2 = ImageOps.exif_transpose(im2)
        if (im2.mode in ("RGBA", "LA")) or ("transparency" in im2.info) or (im2.mode == "P"):
            im2 = im2.convert("RGBA")
        else:
            im2 = im2.convert("RGB")
        return True
    except Exception:
        return False

def reencode_image_rgb(b: bytes, to: str = REENCODE_TO, quality: int = JPEG_QUALITY) -> Tuple[bytes, str]:
    """
    Re-encode with EXIF handling and alpha flattening if needed.
    """
    im = Image.open(io.BytesIO(b))
    im = ImageOps.exif_transpose(im)
    has_alpha = (im.mode in ("RGBA", "LA")) or ("transparency" in im.info) or (im.mode == "P")

    if to.lower() == "png":
        im = im.convert("RGBA" if has_alpha else "RGB")
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        return buf.getvalue(), ".png"

    # default: JPEG
    if has_alpha:
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        if im.mode != "RGB":
            im = im.convert("RGB")

    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue(), ".jpg"

def fail_logger(split: str):
    """
    Returns a logger function for failures:
    - state/{split}_failures.jsonl with {url,status,reason}
    - state/{split}_retry.txt with retry-eligible failures only
    """
    logp = Path(STATE_DIR) / f"{split}_failures.jsonl"
    retryp = Path(STATE_DIR) / f"{split}_retry.txt"
    logp.parent.mkdir(parents=True, exist_ok=True)

    def log(url: str, status: Optional[int], reason: Optional[str]):
        with logp.open("ab") as f:
            f.write(_json.dumps({"url": url, "status": status, "reason": reason}))
            f.write(b"\n")
        if (status in RETRIABLE_STATUSES) or (reason in {"timeout"} or (reason or "").startswith("exception:")):
            with retryp.open("a", encoding="utf-8") as f:
                f.write(url + "\n")

    return log

# ----------------------------
# MediaWiki API helpers
# ----------------------------

async def commons_imageinfo_url(session: aiohttp.ClientSession, title: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Use MediaWiki API to get original URL or a large rendered thumbnail (PNG/JPEG).
    Returns (best_url, mime) or None.
    """
    params = {
        "action": "query",
        "redirects": 1,
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "iiurlwidth": 4096,
        "format": "json",
        "origin": "*",
    }
    try:
        async with session.get(MEDIAWIKI_API, params=params, timeout=DOWNLOAD_TIMEOUT) as r:
            if r.status != 200:
                return None
            data = await r.json(content_type=None)
    except Exception:
        return None

    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        ii = page.get("imageinfo", [])
        if not ii:
            continue
        info = ii[0]
        best = info.get("thumburl") or info.get("url")
        mime = info.get("mime")
        if best:
            return best, mime
    return None

# ----------------------------
# Download helpers
# ----------------------------

async def fetch_bytes(session: aiohttp.ClientSession, url: str) -> Tuple[Optional[bytes], Optional[int], Optional[str]]:
    """
    GET with realistic headers, redirects enabled, exponential backoff + jitter.
    Returns (bytes, status, reason) or (None, status, reason).
    Includes Wikimedia fallbacks and http→https upgrade.
    """
    headers = {
        "User-Agent": random.choice(UA_POOL),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en;q=0.8,it;q=0.7",
        "Referer": _origin(url),
        "Origin": _origin(url),
    }
    delay = 0.6
    for attempt in range(1, RETRIES + 1):
        try:
            async with session.get(url, timeout=DOWNLOAD_TIMEOUT, headers=headers, allow_redirects=True) as r:
                st = r.status
                if st == 200:
                    ctype = r.headers.get("Content-Type", "").lower()
                    b = await r.read()
                    if not b:
                        return None, 200, "empty"
                    if "text/html" in ctype or looks_like_html(b):
                        return None, 200, "html"
                    return b, 200, None

                if st == 404:
                    u = urlparse(url)
                    # http→https upgrade for Wikimedia
                    if u.scheme == "http" and u.netloc in WIKIMEDIA_NETLOC:
                        https_url = f"https://{u.netloc}{u.path}"
                        try:
                            async with session.get(https_url, timeout=DOWNLOAD_TIMEOUT, headers=headers, allow_redirects=True) as r2:
                                if r2.status == 200:
                                    b2 = await r2.read()
                                    ctype2 = r2.headers.get("Content-Type", "").lower()
                                    if "text/html" in ctype2 or looks_like_html(b2):
                                        return None, 200, "html"
                                    return b2, 200, None
                                st = r2.status
                        except Exception:
                            pass
                    # Special:FilePath fallback
                    if u.netloc in WIKIMEDIA_NETLOC:
                        fname = _wikimedia_filename_from_url(url)
                        if fname:
                            sp_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(fname)}?download"
                            try:
                                async with session.get(sp_url, timeout=DOWNLOAD_TIMEOUT, headers=headers, allow_redirects=True) as r3:
                                    if r3.status == 200:
                                        b3 = await r3.read()
                                        ctype3 = r3.headers.get("Content-Type", "").lower()
                                        if "text/html" in ctype3 or looks_like_html(b3):
                                            return None, 200, "html"
                                        return b3, 200, None
                            except Exception:
                                pass
                        # MediaWiki API fallback
                        title = commons_title_from_url(url)
                        if title:
                            try:
                                res = await commons_imageinfo_url(session, title)
                                if res:
                                    best_url, _mime = res
                                    async with session.get(best_url, timeout=DOWNLOAD_TIMEOUT, headers=headers, allow_redirects=True) as r4:
                                        if r4.status == 200:
                                            b4 = await r4.read()
                                            ctype4 = r4.headers.get("Content-Type", "").lower()
                                            if "text/html" in ctype4 or looks_like_html(b4):
                                                return None, 200, "html"
                                            return b4, 200, None
                            except Exception:
                                pass
                    return None, 404, "nonretriable"

                if st in NONRETRIABLE_STATUSES:
                    return None, st, "nonretriable"

                await asyncio.sleep(delay + random.random() * 0.5)
                delay = min(delay * 2, 5)

        except asyncio.TimeoutError:
            if attempt == RETRIES:
                return None, None, "timeout"
            await asyncio.sleep(delay + random.random() * 0.5)
            delay = min(delay * 2, 5)
        except Exception as e:
            if attempt == RETRIES:
                return None, None, f"exception:{type(e).__name__}"
            await asyncio.sleep(delay + random.random() * 0.5)
            delay = min(delay * 2, 5)
    return None, None, "exhausted"

# ----------------------------
# Split processing
# ----------------------------

async def process_split(split: str, df: pl.DataFrame, done_set: Set[str]):
    out_split_dir = Path(OUT_DIR) / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    next_idx = detect_next_shard_index(out_split_dir)
    shardmgr = ShardManager(out_split_dir, start_index=next_idx, shard_size=SHARD_SIZE)

    connector = aiohttp.TCPConnector(
        limit=MAX_CONC_DOWNLOADS * 4,
        limit_per_host=PER_HOST_LIMIT,
        ttl_dns_cache=TTL_DNS_CACHE,
        force_close=True,
        enable_cleanup_closed=True,
    )
    timeout = aiohttp.ClientTimeout(total=None)

    bad_host_strikes = defaultdict(int)
    host_blocked: Set[str] = set()

    headers_base = {
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en;q=0.8,it;q=0.7",
    }

    # Interleave URLs by host using normalized url_n
    urls = df.select(pl.col("url_n")).to_series().to_list()
    by_host = defaultdict(list)
    for u in urls:
        by_host[urlparse(u).netloc].append(u)
    hosts = list(by_host.keys())
    random.shuffle(hosts)
    interleaved = []
    more = True
    idx = 0
    while more:
        more = False
        for h in hosts:
            if idx < len(by_host[h]):
                interleaved.append(by_host[h][idx])
                more = True
        idx += 1

    # Rebuild df in interleaved order (join on url_n)
    df = pl.DataFrame({"url_n": interleaved}).join(df, on="url_n", how="left")

    pbar = tqdm(total=df.height, desc=f"{split}: samples", unit="img")
    ok, fail_dl, fail_val = 0, 0, 0
    flog = fail_logger(split)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers_base, trust_env=True) as session:
        sem_global = asyncio.Semaphore(MAX_CONC_DOWNLOADS)

        async def handle_row(row):
            nonlocal ok, fail_dl, fail_val
            # Always use normalized URL
            url: str = row["url_n"]

            if url in done_set:
                pbar.update(1)
                return

            host = urlparse(url).netloc
            if host in host_blocked:
                fail_dl += 1
                flog(url, 0, "host_blocked")
                pbar.update(1)
                return

            async with sem_global:
                raw, st, why = await fetch_bytes(session, url)

            if raw is None:
                if st in NONRETRIABLE_STATUSES:
                    bad_host_strikes[host] += 1
                    if bad_host_strikes[host] >= HOST_HARDFAIL_THRESHOLD:
                        host_blocked.add(host)
                fail_dl += 1
                flog(url, st, why)
                pbar.update(1)
                return

            # If SVG bytes, rasterize to PNG first
            if is_svg_bytes(raw):
                try:
                    raw = rasterize_svg_to_png(raw)
                except Exception:
                    fail_val += 1
                    pbar.update(1)
                    return

            # Validate
            if not is_valid_image_bytes(raw):
                fail_val += 1
                pbar.update(1)
                return

            # Re-encode
            try:
                raw, ext = reencode_image_rgb(raw, to=REENCODE_TO, quality=JPEG_QUALITY)
            except Exception:
                fail_val += 1
                pbar.update(1)
                return

            by_lang = structs_to_langdict(row["langpack_structs"])
            key = stable_key(url)
            meta = {"url": url, "languages": REQUIRED_LANGS, "by_lang": by_lang}
            shardmgr.write_sample(key, raw, meta, img_ext=ext)
            append_done(split, url)
            pbar.update(1)
            ok += 1

        BATCH = 5000
        try:
            for start in range(0, df.height, BATCH):
                sub = df.slice(start, min(BATCH, df.height - start))
                tasks = [handle_row(r) for r in sub.iter_rows(named=True)]
                await asyncio.gather(*tasks)
        finally:
            shardmgr.close()
            pbar.close()

    print(f"[{split}] ok={ok}, fail_download={fail_dl}, fail_validate={fail_val}")

# ----------------------------
# Entry
# ----------------------------

def process_one_split(split: str, retry_only: bool = False):
    print(f"\n=== Processing split: {split} ===")
    tsvs = list_tsvs_for_split(split)
    if not tsvs:
        print(f"[WARN] No TSV files found for split '{split}'. Pattern: {SPLIT_PATTERNS[split]}")
        return
    print(f"Found {len(tsvs)} shard(s) for {split}.")

    print("Scanning & filtering with Polars (lazy/streaming)...")
    df = read_valid_multilingual_samples(tsv_files=tsvs)
    df = df.with_columns(
        pl.col(COL_IMAGE_URL).map_elements(normalize_url).alias("url_n")
    )

    if df.is_empty():
        print(f"[INFO] No multilingual matches for {split}.")
        return

    if retry_only:
        retry_set = load_retry_set(split)
        if not retry_set:
            print(f"[INFO] No retry list found for split '{split}' (state/{split}_retry.txt). Skipping.")
            return
        df = df.filter(pl.col("url_n").is_in(list(retry_set)))
        print(f"[INFO] Retry-only mode: {df.height} URLs to retry in '{split}'.")

    done_set = load_done_set(split)
    print(f"Already done for {split}: {len(done_set)}")
    if done_set:
        df = df.filter(~pl.col("url_n").is_in(list(done_set)))
        print(f"Remaining after resume filter: {df.height}")

    if df.is_empty():
        print(f"[INFO] Nothing to process for {split}.")
        return

    print("Downloading images & writing WebDataset shards...")
    uvloop.install()
    asyncio.run(process_split(split, df, done_set))
    print(f"Finished {split}.")

def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Build WebDataset shards from WIT TSVs with robust downloader.")
    parser.add_argument("--retry-only", action="store_true",
                        help="Process only URLs listed in state/{split}_retry.txt from the previous run.")
    parser.add_argument("--splits", type=str, default="train,val,test",
                        help="Comma-separated list of splits to process (default: train,val,test).")
    args = parser.parse_args()

    print("Config:")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  OUT_DIR:  {OUT_DIR}")
    print(f"  STATE_DIR:{STATE_DIR}")
    print(f"  REQUIRED_LANGS: {REQUIRED_LANGS}")
    print(f"  PREFERRED_ORDER: {PREFERRED_ORDER}")
    print(f"  VALIDITY_MODE: {VALIDITY_MODE}"
          + (f" (REQUIRED_FIELD_FOR_SINGLE={REQUIRED_FIELD_FOR_SINGLE})" if VALIDITY_MODE=='single' else ""))
    print(f"  SHARD_SIZE: {SHARD_SIZE}, MAX_CONC_DOWNLOADS: {MAX_CONC_DOWNLOADS}, PER_HOST_LIMIT: {PER_HOST_LIMIT}")
    print(f"  REENCODE_TO: {REENCODE_TO}, JPEG_QUALITY: {JPEG_QUALITY}, MAX_IMAGE_BYTES: {MAX_IMAGE_BYTES}")
    print(f"  RETRIES: {RETRIES}, DOWNLOAD_TIMEOUT: {DOWNLOAD_TIMEOUT}, TTL_DNS_CACHE: {TTL_DNS_CACHE}")
    print(f"  RETRY_ONLY: {args.retry_only}")
    print(f"  SPLITS: {args.splits}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        process_one_split(split, retry_only=args.retry_only)

    print("\nAll done!")

if __name__ == "__main__":
    main()
