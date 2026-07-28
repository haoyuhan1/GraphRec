"""MovieLens-1M → GRID TFRecords.

Source: the GroupLens ml-1m release.

    ratings.dat   "UserID::MovieID::Rating::Timestamp"
    movies.dat    "MovieID::Title::Genres"   (latin-1, genres are "|"-separated)

Every rating counts as an interaction (no rating threshold). ML-1M is already
20-core on the user side by construction, so only an item-side filter is
applied: drop movies with < 5 total interactions, which makes the result
directly comparable to the 5-core Amazon/LastFM/Yelp benchmarks. Sequences are
then truncated to the most recent 100 interactions.

The item id space is fixed BEFORE truncation, so movies that survive the
5-core but appear only in truncated-away positions still receive an id — this
is what makes the item count 3,416 rather than a smaller post-truncation
number.

Paper statistics: 6,040 users / 3,416 items / 268,867 transition edges.
"""
import collections
import os

from . import common

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
FILES = ["ratings.dat", "movies.dat"]

MIN_ITEM_COUNT = 5
MAX_SEQ_LEN = 100


def download_raw(raw_dir):
    archive = os.path.join(raw_dir, os.path.basename(URL))
    if not all(os.path.exists(os.path.join(raw_dir, f)) for f in FILES):
        common.download(URL, archive)
        # The zip nests everything under ml-1m/; unzip() flattens by basename.
        common.unzip(archive, raw_dir, members=[f"ml-1m/{f}" for f in FILES])
    return [os.path.join(raw_dir, f) for f in FILES]


def prepare(raw_dir, out_dir):
    ratings_path, movies_path = download_raw(raw_dir)

    print("[1/3] reading ratings ...")
    triples = []
    with open(ratings_path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 4:
                continue
            try:
                triples.append((parts[0], parts[1], int(parts[3])))
            except ValueError:
                continue
    print(f"  {len(triples):,} interactions")

    counts = collections.Counter(t[1] for t in triples)
    keep = {i for i, c in counts.items() if c >= MIN_ITEM_COUNT}
    triples = [t for t in triples if t[1] in keep]
    print(f"  after item_count >= {MIN_ITEM_COUNT}: {len(triples):,} interactions, "
          f"{len(keep):,} items")

    by_user = collections.defaultdict(list)
    for uid, mid, ts in triples:
        by_user[uid].append((mid, ts))
    user_seq = {
        u: [mid for mid, _ in sorted(pairs, key=lambda x: x[1])]
        for u, pairs in by_user.items()
    }

    # Item ids are assigned before truncation — see the module docstring.
    item_order = sorted({m for s in user_seq.values() for m in s})

    user_seq = {u: s[-MAX_SEQ_LEN:] for u, s in user_seq.items()}
    sequences = common.finalize(user_seq)
    print(f"  users {len(sequences):,}  items {len(item_order):,}  "
          f"(truncated to the most recent {MAX_SEQ_LEN})")

    print("[2/3] reading movie titles ...")
    item_text = {}
    with open(movies_path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            mid, title, genres = parts[0], parts[1], parts[2]
            if mid not in keep:
                continue
            item_text[mid] = common.clean_ascii(common.join_fields([
                ("Title", title),
                ("Genres", ", ".join(g for g in genres.split("|") if g)),
            ]))
    for mid in set(item_order) - set(item_text):
        item_text[mid] = f"Movie {mid}"

    print("[3/3] writing TFRecords ...")
    return common.write_dataset(out_dir, sequences, item_text, item_order=item_order)
