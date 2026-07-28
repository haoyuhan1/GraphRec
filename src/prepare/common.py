"""Shared plumbing for the dataset preparation scripts.

The per-dataset modules differ only in how they turn a raw dump into
`(sequences, item_text)`. Everything downstream of that — dense ID remapping,
the leave-2-out split, TFRecord sharding, `items.json` / `id_mapping.json` — is
identical across all of them and lives here.

Sequence convention (identical for all 14 datasets):

    training   = seq[:-2]
    evaluation = seq[:-1]
    testing    = seq          (target = seq[-1], previous item = seq[-2])
"""
import gzip
import json
import os
import shutil
import sys
import urllib.request

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

MIN_SEQ_LEN = 3       # leave-2-out needs >=1 training item + 2 held-out items
_TFOPTS = tf.io.TFRecordOptions(compression_type="GZIP")


# ── Text cleaning ─────────────────────────────────────────────────────────────
# Whitespace/control characters that break the TFRecord text field or trip up
# JSON viewers: U+2028 LS, U+2029 PS, U+0085 NEL, U+000B VT, U+000C FF.
_BAD_WS = str.maketrans({
    c: " " for c in [
        "\n", "\r", "\t",
        "\u2028",   # LINE SEPARATOR
        "\u2029",   # PARAGRAPH SEPARATOR
        "\u0085",   # NEXT LINE
        "\u000b",   # VERTICAL TAB
        "\u000c",   # FORM FEED
    ]
})

_UNICODE_FIXUPS = {
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "–": "-", "—": "-",
    "…": "...",
    " ": " ",
}


def clean_ws(s):
    """Collapse control characters to spaces and strip. Keeps non-ASCII text."""
    if s is None or isinstance(s, float):   # float catches pandas NaN
        return ""
    return str(s).translate(_BAD_WS).strip()


def clean_ascii(s):
    """clean_ws plus transliteration to ASCII (e -> e, n -> n, ...).

    Used by the datasets whose downstream consumers went through numpy's
    `.astype(str)` path, which falls back to ASCII and would otherwise mangle
    accented titles.
    """
    import unicodedata

    s = clean_ws(s)
    for u, a in _UNICODE_FIXUPS.items():
        s = s.replace(u, a)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def join_fields(pairs):
    """Render [(label, value), ...] as "Label: value; Label: value", skipping empties."""
    parts = []
    for label, value in pairs:
        value = clean_ws(value)
        if value:
            parts.append(f"{label}: {value}")
    return "; ".join(parts)


# ── Download ──────────────────────────────────────────────────────────────────

def download(url, dest):
    """Fetch `url` to `dest` unless it already exists. Returns `dest`."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  [cached] {dest}")
        return dest
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    print(f"  downloading {url}")
    print(f"          --> {dest}")

    # Carriage-return progress only works on a terminal; when the output is a
    # log file, report at 10% marks instead of emitting thousands of lines.
    tty = sys.stdout.isatty()
    state = {"next_mark": 0}

    def _progress(block_num, block_size, total_size):
        done = block_num * block_size
        mb = done / 1024 / 1024
        if tty:
            if total_size > 0:
                sys.stdout.write(f"\r    {min(done / total_size * 100, 100):5.1f}%  {mb:.1f} MB")
            else:
                sys.stdout.write(f"\r    {mb:.1f} MB")
            sys.stdout.flush()
        elif total_size > 0:
            pct = min(done / total_size * 100, 100)
            if pct >= state["next_mark"]:
                print(f"    {pct:5.1f}%  {mb:.1f} MB", flush=True)
                state["next_mark"] += 10

    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    if tty:
        sys.stdout.write("\n")
    os.replace(tmp, dest)
    return dest


def unzip(archive, dest_dir, members=None):
    """Extract a .zip archive (whole thing, or just `members`) into dest_dir."""
    import zipfile

    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in members or z.namelist():
            target = os.path.join(dest_dir, os.path.basename(name))
            if os.path.exists(target):
                continue
            print(f"  extracting {name}")
            with z.open(name) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    return dest_dir


def open_maybe_gz(path, encoding="utf-8"):
    """Open a text file that may or may not be gzip-compressed."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding=encoding)
    return open(path, "r", encoding=encoding)


def require_manual(raw_dir, files, dataset, instructions):
    """Fail with actionable instructions when a login-gated raw file is absent."""
    missing = [f for f in files if not os.path.exists(os.path.join(raw_dir, f))]
    if not missing:
        return
    raise SystemExit(
        f"\n{dataset}: the raw data requires an account and cannot be downloaded "
        f"automatically.\n"
        f"Missing under {raw_dir}/:\n"
        + "".join(f"  - {f}\n" for f in missing)
        + f"\n{instructions}\n"
    )


# ── Filtering ─────────────────────────────────────────────────────────────────

def dedup_consecutive(seq):
    """[A, A, B, A] -> [A, B, A]. Repeat interactions later in the sequence are
    real signal and are kept; only immediate repeats are collapsed."""
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def first_seen_order(sequences):
    """Item keys in the order they first appear when walking `sequences`.

    Several of the original pipelines assigned item ids this way rather than by
    sorting. The resulting sequences are identical either way, but the id
    assignment is not cosmetic: it decides how `torch.topk` breaks score ties
    and which items random padding draws, so it shifts the reported metrics
    slightly. Use whichever order the dataset's numbers were produced with.
    """
    seen, order = set(), []
    for _user, seq in sequences:
        for item in seq:
            if item not in seen:
                seen.add(item)
                order.append(item)
    return order


def kcore_one_pass(user_seq, k=5, min_seq_len=MIN_SEQ_LEN, redup=True):
    """One-pass k-core: drop items seen < k times, then drop users left too short.

    Deliberately NOT iterated — dropping mid-sequence items can push an item
    below k again, but re-running would change the published statistics.
    `redup` re-collapses consecutive duplicates exposed by item removal.
    """
    import collections

    counts = collections.Counter(i for s in user_seq.values() for i in s)
    keep = {i for i, c in counts.items() if c >= k}
    out = {}
    for u, s in user_seq.items():
        s = [i for i in s if i in keep]
        if redup:
            s = dedup_consecutive(s)
        if len(s) >= max(k, min_seq_len):
            out[u] = s
    return out


def bipartite_kcore(interactions, min_item_users, min_user_items, iterative=False):
    """k-core on the (user, item) bipartite graph over [(user, item, ts), ...].

    Drops items touched by fewer than `min_item_users` distinct users, then
    users left with fewer than `min_user_items` distinct items.

    `iterative=False` runs that once — the Amazon convention, which tolerates a
    few items falling below the threshold after the user-side pass. `iterative`
    repeats to a fixed point, which is what the LastFM benchmark numbers were
    produced with. The two give different item counts, so this is per-dataset
    and not a free choice.
    """
    import collections

    if min_item_users <= 1 and min_user_items <= 1:
        return interactions

    while True:
        n_before = len(interactions)

        item_users = collections.defaultdict(set)
        for u, i, _ in interactions:
            item_users[i].add(u)
        keep_items = {i for i, us in item_users.items() if len(us) >= min_item_users}
        interactions = [t for t in interactions if t[1] in keep_items]

        user_items = collections.defaultdict(set)
        for u, i, _ in interactions:
            user_items[u].add(i)
        keep_users = {u for u, its in user_items.items() if len(its) >= min_user_items}
        interactions = [t for t in interactions if t[0] in keep_users]

        if not iterative or len(interactions) == n_before:
            return interactions


# ── Tag-derived item text (Delicious / LastFM) ────────────────────────────────

def valid_tag_ids(tag_users, tag_names, min_users=5, max_len=40):
    """Tags worth putting in an item's text.

    Counting distinct USERS rather than taggings is the main noise filter: it
    removes personal tags that a single power-user sprayed across many items.
    Tags that are over-long or contain no letters ("?????", "---") also go.
    """
    def _ok(name):
        name = (name or "").strip()
        return bool(name) and len(name) <= max_len and any(c.isalpha() for c in name)

    return {
        tid for tid, users in tag_users.items()
        if len(users) >= min_users and _ok(tag_names.get(tid, ""))
    }


def top_tag_names(tag_counter, valid, tag_names, top_k=5, min_count=1):
    """The `top_k` most-applied valid tag names for one item, count-desc."""
    if not tag_counter:
        return []
    kept = [
        (tid, c) for tid, c in tag_counter.items()
        if c >= min_count and tid in valid
    ]
    kept.sort(key=lambda x: (-x[1], x[0]))
    names = []
    for tid, _ in kept[:top_k]:
        name = clean_ascii(tag_names.get(tid, "")).strip()
        if name:
            names.append(name)
    return names


# ── TFRecord writing ──────────────────────────────────────────────────────────

def _int64(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def write_dataset(out_dir, sequences, item_text, item_order=None,
                  n_user_parts=32, n_item_files=16, with_text=False):
    """Write the full GRID layout for one dataset.

    Args:
      out_dir:    destination `<data_root>/<name>/`.
      sequences:  ordered [(user_key, [item_key, ...]), ...]. Position in this
                  list becomes the 0-indexed user_id, so the caller controls
                  user ordering.
      item_text:  {item_key: text}. Every item key in `sequences` must appear.
      item_order: explicit item-key order defining the 0-indexed item ids.
                  Defaults to sorted(); pass first-seen order to match a source
                  pipeline that assigned ids that way.
      with_text:  also embed the per-interaction item text in the split records.
                  The GRID trainer consumed this field; TGH reads only
                  `user_id` / `sequence_data`, and including it inflates the
                  dataset by 100-1000x. Off by default.

    Returns a stats dict.
    """
    if item_order is None:
        item_order = sorted({i for _, s in sequences for i in s})
    item2id = {key: i for i, key in enumerate(item_order)}

    missing = next((i for _, s in sequences for i in s if i not in item2id), None)
    if missing is not None:
        raise KeyError(f"item {missing!r} appears in a sequence but not in item_order")

    os.makedirs(out_dir, exist_ok=True)

    # items/
    items_dir = os.path.join(out_dir, "items")
    os.makedirs(items_dir, exist_ok=True)
    chunk = (len(item_order) + n_item_files - 1) // n_item_files
    for fi in range(n_item_files):
        part = item_order[fi * chunk:(fi + 1) * chunk]
        if not part and fi:
            continue
        path = os.path.join(items_dir, f"data_{fi}.tfrecord.gz")
        with tf.io.TFRecordWriter(path, options=_TFOPTS) as w:
            for key in part:
                ex = tf.train.Example(features=tf.train.Features(feature={
                    "id":   _int64([item2id[key]]),
                    "text": _bytes([item_text[key].encode("utf-8")]),
                }))
                w.write(ex.SerializeToString())

    with open(os.path.join(out_dir, "items.json"), "w", encoding="utf-8") as f:
        json.dump(
            [{"id": item2id[k], "text": item_text[k]} for k in item_order],
            f, ensure_ascii=False,
        )

    # training / evaluation / testing
    splits = {
        "training":   lambda s: s[:-2],
        "evaluation": lambda s: s[:-1],
        "testing":    lambda s: s,
    }
    for split in splits:
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)

    n_users = len(sequences)
    chunk_u = (n_users + n_user_parts - 1) // n_user_parts
    for fi in range(n_user_parts):
        lo, hi = fi * chunk_u, (fi + 1) * chunk_u
        if lo >= n_users:
            break
        part = sequences[lo:hi]
        for split, slicer in splits.items():
            path = os.path.join(out_dir, split, f"partition_{fi}.tfrecord.gz")
            with tf.io.TFRecordWriter(path, options=_TFOPTS) as w:
                for uid, (_key, seq) in enumerate(part, start=lo):
                    sub = slicer(seq)
                    feature = {
                        "user_id":       _int64([uid]),
                        "sequence_data": _int64([item2id[i] for i in sub]),
                    }
                    if with_text:
                        feature["text"] = _bytes(
                            [item_text[i].encode("utf-8") for i in sub]
                        )
                    ex = tf.train.Example(features=tf.train.Features(feature=feature))
                    w.write(ex.SerializeToString())

    n_inter = sum(len(s) for _, s in sequences)
    with open(os.path.join(out_dir, "id_mapping.json"), "w") as f:
        json.dump({
            "item2id": {str(k): v for k, v in item2id.items()},
            "n_items": len(item2id),
            "n_users": n_users,
        }, f)

    stats = {
        "users": n_users,
        "items": len(item2id),
        "interactions": n_inter,
        "avg_seq_len": n_inter / n_users if n_users else 0.0,
    }
    print(f"\n  wrote {out_dir}/")
    print(f"    users        : {stats['users']:,}")
    print(f"    items        : {stats['items']:,}")
    print(f"    interactions : {stats['interactions']:,}")
    print(f"    avg seq len  : {stats['avg_seq_len']:.2f}")
    return stats


def finalize(user_seq, min_seq_len=MIN_SEQ_LEN, sort_users=True):
    """Drop too-short users and return the ordered `sequences` list.

    `sort_users=True` orders by user key so the assigned user ids are
    reproducible across runs regardless of dict insertion order.
    """
    items = [(u, s) for u, s in user_seq.items() if len(s) >= min_seq_len]
    if sort_users:
        items.sort(key=lambda kv: kv[0])
    return items
