"""Yelp → GRID TFRecords.

Source: the canonical Yelp sequential-recommendation benchmark as released by
LETTER, which is the same split used by TIGER / CoFiRec (Yelp reviews from 2019
onwards, iterative 5-core, no truncation):

    data/Yelp/Yelp.inter.json   {user_id: [item_id, ...]}  already 0-indexed
    data/Yelp/Yelp.item.json    {item_id: {"title", "description"}}

`description` holds the business categories, so the item text reproduces the
"Title: ...; Categories: ..." schema. Location / stars / attribute flags are
deliberately left out: benchmark papers use only (name, categories), and the
dropped fields are boolean flags shared by ~90% of businesses, which cost T5
tokens without adding discriminative signal.

Paper statistics: 30,431 users / 20,033 items / 219,632 transition edges /
316,354 interactions — reproduced exactly by these two files.
"""
import json
import os

from . import common

BASE_URL = "https://raw.githubusercontent.com/HonghuiBao2000/LETTER/master/data/Yelp"
FILES = ["Yelp.inter.json", "Yelp.item.json"]


def download_raw(raw_dir):
    return [common.download(f"{BASE_URL}/{f}", os.path.join(raw_dir, f)) for f in FILES]


def prepare(raw_dir, out_dir):
    inter_path, item_path = download_raw(raw_dir)

    print("[1/2] reading sequences and item text ...")
    with open(inter_path) as f:
        inter = json.load(f)
    with open(item_path) as f:
        items = json.load(f)

    user_seq = {int(u): [int(i) for i in seq] for u, seq in inter.items()}

    item_text = {}
    for iid, rec in items.items():
        item_text[int(iid)] = common.clean_ascii(common.join_fields([
            ("Title", rec.get("title", "")),
            ("Categories", rec.get("description", "")),
        ]))

    sequences = common.finalize(user_seq)
    used = {i for _, s in sequences for i in s}
    for i in used - set(item_text):
        item_text[i] = f"Item: {i}"
    print(f"  users {len(sequences):,}  items {len(used):,}")

    print("[2/2] writing TFRecords ...")
    return common.write_dataset(out_dir, sequences, item_text)
