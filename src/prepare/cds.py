"""Amazon CDs_and_Vinyl (5-core) → GRID TFRecords.

Source: the Amazon Product Data 2014 dump hosted by SNAP. The download and
sequence-construction logic follows ActionPiece's `AmazonReviews2014` dataset
(genrec/datasets/AmazonReviews2014), which is also what the Beauty/Sports/Toys
splits were built from.

    reviews_CDs_and_Vinyl_5.json.gz   interactions (already 5-core at source)
    meta_CDs_and_Vinyl.json.gz        item metadata for the text field

Both files are Python dict literals, not JSON, so they are parsed with
`ast.literal_eval` (the reviews file additionally survives `json.loads` once
true/false are rewritten).

Paper statistics: 75,258 users / 64,443 items / 810,347 transition edges.
"""
import ast
import collections
import gzip
import json
import os

from . import common

CATEGORY = "CDs_and_Vinyl"
BASE_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
REVIEWS_URL = f"{BASE_URL}reviews_{CATEGORY}_5.json.gz"
META_URL = f"{BASE_URL}meta_{CATEGORY}.json.gz"


def _parse_reviews(path):
    """Yield review dicts. The reviews file is valid JSON once the Python
    literals true/false are rewritten."""
    with gzip.open(path, "rb") as f:
        for line in f:
            line = line.replace(b"true", b"True").replace(b"false", b"False")
            try:
                yield ast.literal_eval(line.decode("utf-8", errors="replace"))
            except (ValueError, SyntaxError):
                continue


def _parse_meta(path):
    """Yield item-metadata dicts (Python dict literals, single-quoted)."""
    with gzip.open(path, "rb") as f:
        for line in f:
            try:
                yield ast.literal_eval(line.decode("utf-8", errors="replace"))
            except (ValueError, SyntaxError):
                continue


def _item_text(meta):
    """Match the text schema used for the GRID Beauty/Sports/Toys items."""
    parts = []
    title = meta.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(f"Title: {common.clean_ws(title)}")

    brand = meta.get("brand")
    if isinstance(brand, str) and brand.strip():
        parts.append(f"Brand: {common.clean_ws(brand)}")

    cats = meta.get("categories") or []
    if cats:
        # Amazon 2014 stores a list of category paths; keep the first path.
        flat = cats[0] if isinstance(cats[0], list) else cats
        if flat:
            # Rendered as a Python list repr — kept verbatim because the
            # released item embeddings were generated from this exact string.
            parts.append(f"Categories: {flat}")

    price = meta.get("price")
    if price:
        if isinstance(price, str):
            stripped = price.replace("$", "").strip()
            try:
                stripped = str(float(stripped))
            except ValueError:
                pass
            parts.append(f"Price: {stripped}")
        else:
            parts.append(f"Price: {price}")

    if not parts:
        return f"ASIN: {meta.get('asin', 'unknown')}"
    return "; ".join(parts) + ";"


def download_raw(raw_dir):
    return (
        common.download(REVIEWS_URL, os.path.join(raw_dir, os.path.basename(REVIEWS_URL))),
        common.download(META_URL, os.path.join(raw_dir, os.path.basename(META_URL))),
    )


def prepare(raw_dir, out_dir):
    reviews_path, meta_path = download_raw(raw_dir)

    print("[1/3] reading reviews ...")
    cache = os.path.join(raw_dir, "raw_item_seqs.json")
    if os.path.exists(cache):
        print(f"  [cached] {cache}")
        with open(cache) as f:
            raw_seqs = json.load(f)
    else:
        by_user = collections.defaultdict(list)
        n = 0
        for r in _parse_reviews(reviews_path):
            try:
                by_user[r["reviewerID"]].append((r["asin"], int(r["unixReviewTime"])))
            except KeyError:
                continue
            n += 1
        raw_seqs = {
            u: [asin for asin, _ in sorted(items, key=lambda x: x[1])]
            for u, items in by_user.items()
        }
        with open(cache, "w") as f:
            json.dump(raw_seqs, f)
        print(f"  {n:,} interactions over {len(raw_seqs):,} users")

    sequences = common.finalize(raw_seqs)
    used = {a for _, s in sequences for a in s}
    print(f"  users kept (len>={common.MIN_SEQ_LEN}): {len(sequences):,}")

    print("[2/3] reading item metadata ...")
    item_text = {}
    for meta in _parse_meta(meta_path):
        asin = meta.get("asin")
        if asin in used and asin not in item_text:
            item_text[asin] = _item_text(meta)
    # Items with no metadata row still need a text field for the encoder.
    for asin in used - set(item_text):
        item_text[asin] = f"ASIN: {asin}"
    print(f"  {len(item_text):,} items")

    print("[3/3] writing TFRecords ...")
    return common.write_dataset(out_dir, sequences, item_text)
