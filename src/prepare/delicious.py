"""Delicious-2K (HetRec 2011) → GRID TFRecords.

Source: the HetRec 2011 Delicious-2K release from GroupLens.

    user_taggedbookmarks.dat   tagging events with day/month/year/h/m/s columns
    bookmarks.dat              bookmark id, title, url
    tags.dat                   tag id -> tag name

Bookmarks are the items. Delicious has no explicit interaction log, so the
sequence comes from tagging events: each (user, bookmark) pair is one
interaction timestamped at its first tagging, and the user's sequence is those
pairs in time order.

Filtering is a ONE-PASS bipartite 5-core — drop bookmarks tagged by < 5 distinct
users, then drop users left with < 5 distinct bookmarks. Not iterated, matching
the Amazon convention. Sequences are not truncated.

Paper statistics: 718 users / 1,200 items / 4,016 transition edges.
"""
import collections
import os

from . import common

URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-delicious-2k.zip"
FILES = ["user_taggedbookmarks.dat", "bookmarks.dat", "tags.dat"]

MIN_ITEM_USERS = 5
MIN_USER_ITEMS = 5
TOP_TAGS = 5
MIN_GLOBAL_TAG_USERS = 5
MAX_TAG_LEN = 40


def download_raw(raw_dir):
    archive = os.path.join(raw_dir, os.path.basename(URL))
    if not all(os.path.exists(os.path.join(raw_dir, f)) for f in FILES):
        common.download(URL, archive)
        common.unzip(archive, raw_dir, members=FILES)
    return [os.path.join(raw_dir, f) for f in FILES]


def _parse_tagged(path):
    """Return (interactions, per-bookmark tag counts, tag -> distinct users).

    A (user, bookmark) pair is recorded once, at its earliest tagging event.
    """
    interactions, seen = [], set()
    tag_counts = collections.defaultdict(collections.Counter)
    tag_users = collections.defaultdict(set)
    with open(path, encoding="utf-8", errors="replace") as f:
        f.readline()                                    # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            uid, bid, tid = parts[0], parts[1], parts[2]
            try:
                # day, month, year, hour, minute, second -> sortable integer
                ts = (int(parts[5]) * 10_000_000_000 + int(parts[4]) * 100_000_000
                      + int(parts[3]) * 1_000_000 + int(parts[6]) * 10_000
                      + int(parts[7]) * 100 + int(parts[8]))
            except (ValueError, IndexError):
                ts = 0
            if (uid, bid) not in seen:
                seen.add((uid, bid))
                interactions.append((uid, bid, ts))
            if tid:
                tag_counts[bid][tid] += 1
                tag_users[tid].add(uid)
    return interactions, tag_counts, tag_users


def _parse_tsv(path, key_col, val_cols):
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        f.readline()                                    # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(key_col, *val_cols):
                continue
            out[parts[key_col]] = [parts[c] for c in val_cols]
    return out


def prepare(raw_dir, out_dir):
    tagged_path, bookmarks_path, tags_path = download_raw(raw_dir)

    print("[1/3] reading tagging events ...")
    interactions, tag_counts, tag_users = _parse_tagged(tagged_path)
    print(f"  {len(interactions):,} distinct (user, bookmark) interactions")

    print(f"[2/3] one-pass {MIN_ITEM_USERS}-core ...")
    interactions = common.bipartite_kcore(
        interactions, MIN_ITEM_USERS, MIN_USER_ITEMS, iterative=False,
    )
    print(f"  {len(interactions):,} interactions remain")

    by_user = collections.defaultdict(list)
    for uid, bid, ts in interactions:
        by_user[uid].append((bid, ts))
    user_seq = {
        u: [bid for bid, _ in sorted(pairs, key=lambda x: x[1])]
        for u, pairs in by_user.items()
    }

    # User and item ids follow FIRST-APPEARANCE order in the tagging file, not
    # sorted order. Both orderings produce the same sequences, but the id
    # assignment feeds torch.topk's tie-breaking and the random-padding draw in
    # tgh.py, which moves Recall by about one user here. First-appearance is
    # what the reported numbers were produced with.
    sequences = common.finalize(user_seq, sort_users=False)
    used = common.first_seen_order(sequences)
    print(f"  users {len(sequences):,}  items {len(used):,}")

    # Item text: "Title: ...; Tags: t1, t2, ..."
    bookmarks = _parse_tsv(bookmarks_path, 0, [2])      # id -> [title]
    tag_names = {k: v[0].strip() for k, v in _parse_tsv(tags_path, 0, [1]).items()}
    valid = common.valid_tag_ids(
        tag_users, tag_names, MIN_GLOBAL_TAG_USERS, MAX_TAG_LEN,
    )
    print(f"  {len(valid):,}/{len(tag_users):,} tags survive the noise filter")

    item_text = {}
    for bid in used:
        title = common.clean_ascii((bookmarks.get(bid) or [""])[0])
        names = common.top_tag_names(tag_counts.get(bid), valid, tag_names, TOP_TAGS)
        text = common.join_fields([("Title", title), ("Tags", ", ".join(names))])
        item_text[bid] = text or f"Bookmark {bid}"

    print("[3/3] writing TFRecords ...")
    return common.write_dataset(out_dir, sequences, item_text)
