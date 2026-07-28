"""Last.fm-2K (HetRec 2011) → GRID TFRecords.

Source: the HetRec 2011 Last.fm-2K release from GroupLens.

    user_taggedartists-timestamps.dat   "userID  artistID  tagID  timestamp(ms)"
    artists.dat                         "id  name  url  pictureURL"
    tags.dat                            "tagID  tagValue"

Artists are the items. `user_artists.dat` holds play counts but no timestamps,
so it cannot seed a chronological sequence; the tagging-event file is used
instead, which is the standard choice in the HetRec-LastFM sequential
recommendation literature. Each (user, artist) pair is kept once at its
EARLIEST tagging timestamp.

The 5-core here is ITERATIVE, unlike Delicious's single pass: dropping
short-sequence users can push artists below 5 taggers, and dropping those
artists can in turn push users below the length floor. Iterating to a fixed
point is what reproduces the published 1,090 / 3,646 figures.

Sequences are truncated to the most recent 100 interactions, with item ids
assigned before truncation (see ml1m for the same convention).

Paper statistics: 1,090 users / 3,646 items / 30,372 transition edges.
"""
import collections
import os

from . import common

URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
FILES = ["user_taggedartists-timestamps.dat", "artists.dat", "tags.dat"]

MIN_ITEM_USERS = 5
MIN_USER_ITEMS = 5      # both sides are 5 here — see the docstring on iteration
MAX_SEQ_LEN = 100
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
    """Return (interactions, per-artist tag counts, tag -> distinct users)."""
    earliest = {}
    tag_counts = collections.defaultdict(collections.Counter)
    tag_users = collections.defaultdict(set)
    skipped = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        f.readline()                                    # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                skipped += 1
                continue
            uid, aid, tid = parts[0], parts[1], parts[2]
            try:
                ts = int(parts[3])
            except ValueError:
                skipped += 1
                continue
            # A few HetRec rows carry ts=0; they are kept and sort to the front
            # of that user's sequence.
            key = (uid, aid)
            if key not in earliest or ts < earliest[key]:
                earliest[key] = ts
            tag_counts[aid][tid] += 1
            tag_users[tid].add(uid)
    if skipped:
        print(f"  (skipped {skipped:,} malformed rows)")
    return [(u, a, ts) for (u, a), ts in earliest.items()], tag_counts, tag_users


def _parse_tsv(path, key_col, val_col):
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        f.readline()                                    # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > max(key_col, val_col):
                out[parts[key_col]] = parts[val_col]
    return out


def prepare(raw_dir, out_dir):
    tagged_path, artists_path, tags_path = download_raw(raw_dir)

    print("[1/3] reading tagging events ...")
    interactions, tag_counts, tag_users = _parse_tagged(tagged_path)
    print(f"  {len(interactions):,} distinct (user, artist) interactions")

    print(f"[2/3] iterative {MIN_ITEM_USERS}-core ...")
    interactions = common.bipartite_kcore(
        interactions, MIN_ITEM_USERS, MIN_USER_ITEMS, iterative=True,
    )
    print(f"  {len(interactions):,} interactions remain")

    by_user = collections.defaultdict(list)
    for uid, aid, ts in interactions:
        by_user[uid].append((aid, ts))
    user_seq = {
        u: [aid for aid, _ in sorted(pairs, key=lambda x: x[1])]
        for u, pairs in by_user.items()
    }

    item_order = sorted({a for s in user_seq.values() for a in s})
    user_seq = {u: s[-MAX_SEQ_LEN:] for u, s in user_seq.items()}
    sequences = common.finalize(user_seq)
    print(f"  users {len(sequences):,}  items {len(item_order):,}")

    artists = _parse_tsv(artists_path, 0, 1)
    tag_names = {k: v.strip() for k, v in _parse_tsv(tags_path, 0, 1).items()}
    valid = common.valid_tag_ids(
        tag_users, tag_names, MIN_GLOBAL_TAG_USERS, MAX_TAG_LEN,
    )
    print(f"  {len(valid):,}/{len(tag_users):,} tags survive the noise filter")

    item_text = {}
    for aid in item_order:
        name = common.clean_ascii(artists.get(aid, ""))
        names = common.top_tag_names(tag_counts.get(aid), valid, tag_names, TOP_TAGS)
        text = common.join_fields([("Artist", name), ("Tags", ", ".join(names))])
        item_text[aid] = text or f"Artist {aid}"

    print("[3/3] writing TFRecords ...")
    return common.write_dataset(out_dir, sequences, item_text, item_order=item_order)
