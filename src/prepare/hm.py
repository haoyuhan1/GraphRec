"""H&M Personalized Fashion Recommendations → GRID TFRecords.

Source: the Kaggle competition dataset. It is behind a competition-rules accept
and cannot be downloaded without credentials, so the two CSVs must be placed in
the raw directory by hand (or fetched with the Kaggle CLI).

    articles.csv            article metadata
    transactions_train.csv  ~31.8M purchases, 2018-09-20 .. 2020-09-22

Sequence construction: group by customer, sort by (t_dat, original CSV row) so
same-day purchases keep a deterministic order, then collapse consecutive
duplicates. No k-core — only the structural len >= 3 requirement.

Paper statistics: 1,077,045 users / 104,468 items / 19,487,762 transition edges.
"""
import os

from . import common

RAW_FILES = ["articles.csv", "transactions_train.csv"]

_MANUAL = """\
Download the two CSVs from the competition page (accepting the rules first):

    https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data

or with the Kaggle CLI:

    kaggle competitions download -c h-and-m-personalized-fashion-recommendations \\
        -f articles.csv -f transactions_train.csv

then unzip them into the raw directory shown above.\
"""

TEXT_COLS = [
    ("Title",       "prod_name"),
    ("Type",        "product_type_name"),
    ("Group",       "product_group_name"),
    ("Color",       "colour_group_name"),
    ("Department",  "department_name"),
    ("Description", "detail_desc"),
]


def download_raw(raw_dir):
    common.require_manual(raw_dir, RAW_FILES, "H&M", _MANUAL)
    return [os.path.join(raw_dir, f) for f in RAW_FILES]


def prepare(raw_dir, out_dir):
    import pandas as pd

    articles_path, tx_path = download_raw(raw_dir)

    print("[1/3] reading articles ...")
    art = pd.read_csv(
        articles_path,
        usecols=["article_id"] + [c for _, c in TEXT_COLS],
        dtype=str,
        keep_default_na=False,
    )
    item_text = {}
    for row in art.itertuples(index=False):
        text = common.join_fields(
            [(label, getattr(row, col)) for label, col in TEXT_COLS]
        )
        if text:
            item_text[row.article_id] = text
    del art
    print(f"  articles with usable text: {len(item_text):,}")

    print("[2/3] reading transactions (this needs a few GB of RAM) ...")
    tx = pd.read_csv(
        tx_path,
        usecols=["t_dat", "customer_id", "article_id"],
        dtype={"customer_id": str, "article_id": str},
        parse_dates=["t_dat"],
    )
    print(f"  rows: {len(tx):,}")
    tx = tx.reset_index().rename(columns={"index": "orig_idx"})
    tx = tx.sort_values(
        ["customer_id", "t_dat", "orig_idx"], kind="mergesort"
    ).reset_index(drop=True)

    user_seq = {}
    cur_uid, cur_seq = None, []
    for cid, article_id in zip(tx["customer_id"].values, tx["article_id"].values):
        if cid != cur_uid:
            if cur_uid is not None:
                user_seq[cur_uid] = cur_seq
            cur_uid, cur_seq = cid, []
        if article_id not in item_text:
            continue
        if cur_seq and cur_seq[-1] == article_id:   # collapse consecutive dups
            continue
        cur_seq.append(article_id)
    if cur_uid is not None:
        user_seq[cur_uid] = cur_seq
    del tx

    sequences = common.finalize(user_seq)
    used = {i for _, s in sequences for i in s}
    print(f"  users {len(sequences):,}  items {len(used):,}")

    print("[3/3] writing TFRecords ...")
    return common.write_dataset(
        out_dir, sequences, item_text, n_user_parts=64, n_item_files=32,
    )
