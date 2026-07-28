"""Goodreads Comics-Graphic and Children → GRID TFRecords.

Source: the UCSD Goodreads dump (Wan & McAuley). Per genre:

    goodreads_books_<genre>.json.gz           book metadata
    goodreads_interactions_<genre>.json.gz     user-book interactions
    goodreads_book_authors.json.gz             shared author-id -> name table

Filtering:
  - rating >= 4 only (standard binarisation of implicit positive feedback).
  - time-ordered by `date_added`.
  - consecutive duplicates collapsed; a book re-read later in the sequence is
    kept as real repeat signal.
  - one-pass 5-core: drop books seen < 5 times, then drop users left shorter
    than 5. Not iterated — see common.kcore_one_pass.

Paper statistics:
    Comics   :  89,186 users /  48,623 items / 1,282,693 transition edges
    Children : 163,143 users /  55,221 items / 1,622,817 transition edges
"""
import collections
import datetime as dt
import gzip
import json
import os

from . import common

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads"
AUTHORS_FILE = "goodreads_book_authors.json.gz"

# dataset name -> (books file, interactions file)
GENRES = {
    "goodreads-comics": (
        "goodreads_books_comics_graphic.json.gz",
        "goodreads_interactions_comics_graphic.json.gz",
    ),
    "goodreads-children": (
        "goodreads_books_children.json.gz",
        "goodreads_interactions_children.json.gz",
    ),
}

RATING_MIN = 4
KCORE = 5


def download_raw(raw_dir, dataset):
    books_file, inter_file = GENRES[dataset]
    return (
        common.download(f"{BASE_URL}/byGenre/{books_file}", os.path.join(raw_dir, books_file)),
        common.download(f"{BASE_URL}/byGenre/{inter_file}", os.path.join(raw_dir, inter_file)),
        common.download(f"{BASE_URL}/{AUTHORS_FILE}", os.path.join(raw_dir, AUTHORS_FILE)),
    )


def _load_authors(path):
    authors = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            aid, name = r.get("author_id"), common.clean_ws(r.get("name", ""))
            if aid and name:
                authors[aid] = name
    return authors


def _book_text(book, authors):
    names = [
        authors[a["author_id"]]
        for a in (book.get("authors") or [])
        if a.get("author_id") in authors
    ]
    return common.join_fields([
        ("Title", book.get("title", "")),
        ("Authors", ", ".join(names)),
        ("Publisher", book.get("publisher", "")),
        ("Year", book.get("publication_year", "")),
        ("Description", book.get("description", "")),
    ])


def prepare(raw_dir, out_dir, dataset=None):
    if dataset is None:
        dataset = os.path.basename(out_dir.rstrip("/"))
    if dataset not in GENRES:
        raise ValueError(f"{dataset!r} is not a Goodreads genre; expected one of {sorted(GENRES)}")

    books_path, inter_path, authors_path = download_raw(raw_dir, dataset)

    print("[1/4] reading authors ...")
    authors = _load_authors(authors_path)
    print(f"  {len(authors):,} authors")

    print("[2/4] reading books ...")
    item_text = {}
    with gzip.open(books_path, "rt") as f:
        for line in f:
            book = json.loads(line)
            bid = book.get("book_id")
            text = _book_text(book, authors)
            if bid and text:
                item_text[bid] = text
    print(f"  books with usable text: {len(item_text):,}")

    print(f"[3/4] reading interactions (rating >= {RATING_MIN}) ...")
    records = []
    with gzip.open(inter_path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if int(r.get("rating", 0)) < RATING_MIN:
                continue
            bid = r.get("book_id")
            if bid not in item_text:
                continue
            try:
                ts = dt.datetime.strptime(r["date_added"], "%a %b %d %H:%M:%S %z %Y")
            except (KeyError, ValueError):
                continue
            records.append((r["user_id"], ts, bid))
    print(f"  kept: {len(records):,}")

    records.sort(key=lambda x: (x[0], x[1]))
    user_seq = collections.defaultdict(list)
    for uid, _ts, bid in records:
        seq = user_seq[uid]
        if not seq or seq[-1] != bid:      # collapse consecutive duplicates
            seq.append(bid)
    del records

    user_seq = common.kcore_one_pass(user_seq, k=KCORE)
    sequences = common.finalize(user_seq)
    used = {b for _, s in sequences for b in s}
    print(f"  after {KCORE}-core: users {len(sequences):,}  items {len(used):,}")

    print("[4/4] writing TFRecords ...")
    return common.write_dataset(out_dir, sequences, item_text)
