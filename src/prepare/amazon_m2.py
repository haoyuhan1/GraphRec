"""Amazon-M2 (UK locale) → GRID TFRecords.

Source: the KDD Cup 2023 Amazon-M2 multilingual session dataset. It is behind a
challenge login and cannot be downloaded automatically, so the two CSVs must be
placed in the raw directory by hand.

    products_train.csv   product catalogue, all locales
    sessions_train.csv   browsing sessions, all locales

Only the UK locale is used. Each SESSION is treated as one "user": the sequence
is `prev_items ++ [next_item]`, which is the natural unit here because Amazon-M2
ships anonymous sessions rather than persistent user ids. Only the test-set
sessions come from a different time period, so `sessions_test_*` is not used.
No k-core is applied — only the structural len >= 3 requirement.

Paper statistics: 1,182,181 sessions / 494,409 items / 1,500,196 transition edges.
"""
import os

from . import common

LOCALE = "UK"
RAW_FILES = ["products_train.csv", "sessions_train.csv"]

_MANUAL = """\
Download the Amazon-M2 KDD Cup 2023 data (free account required):

    https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge

and place products_train.csv and sessions_train.csv in the raw directory above.\
"""

TEXT_COLS = [
    ("Title",    "title"),
    ("Brand",    "brand"),
    ("Color",    "color"),
    ("Material", "material"),
]


def download_raw(raw_dir):
    common.require_manual(raw_dir, RAW_FILES, "Amazon-M2", _MANUAL)
    return [os.path.join(raw_dir, f) for f in RAW_FILES]


def _parse_prev_items(s):
    """"['A' 'B' 'C']" -> ['A', 'B', 'C'] (the field may contain newlines)."""
    return s.strip("[]").replace("\n", " ").replace("'", " ").split()


def prepare(raw_dir, out_dir):
    import pandas as pd

    products_path, sessions_path = download_raw(raw_dir)

    print(f"[1/3] reading products (locale={LOCALE}) ...")
    prod = pd.read_csv(
        products_path,
        usecols=["id", "locale", "title", "brand", "color", "material", "desc"],
        dtype=str,
        keep_default_na=False,
    )
    prod = prod[prod["locale"] == LOCALE]
    prod = prod[prod["title"] != ""].drop_duplicates(subset=["id"])
    print(f"  {LOCALE} products with a title: {len(prod):,}")

    item_text = {}
    for row in prod.itertuples(index=False):
        item_text[row.id] = common.join_fields(
            [(label, getattr(row, col)) for label, col in TEXT_COLS]
            + [("Description", row.desc)]
        )
    del prod

    print(f"[2/3] reading sessions (locale={LOCALE}) ...")
    sess = pd.read_csv(
        sessions_path,
        usecols=["prev_items", "next_item", "locale"],
        dtype=str,
        keep_default_na=False,
    )
    sess = sess[sess["locale"] == LOCALE]
    print(f"  {LOCALE} sessions: {len(sess):,}")

    # Session index is the user key, so ordering follows the original CSV.
    user_seq = {}
    for idx, (prev, nxt) in enumerate(zip(sess["prev_items"], sess["next_item"])):
        seq = [i for i in _parse_prev_items(prev) + [nxt] if i in item_text]
        if seq:
            user_seq[idx] = seq
    del sess

    sequences = common.finalize(user_seq)
    used = {i for _, s in sequences for i in s}
    print(f"  sessions kept {len(sequences):,}  items {len(used):,}")

    print("[3/3] writing TFRecords ...")
    return common.write_dataset(
        out_dir, sequences, item_text, n_user_parts=64, n_item_files=32,
    )
