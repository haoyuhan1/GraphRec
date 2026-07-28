"""Steam (UCSD/McAuley dump) → GRID TFRecords.

Source: the Steam game and review dumps mirrored alongside the SASRec release.

    steam_games.json.gz     game metadata
    steam_reviews.json.gz   one review per line

Filtering follows SASRec's preprocessing: every review is a positive
interaction (no play-hours threshold), and the 5-core is applied from the
ORIGINAL user/item frequencies in a single pass — a record is kept iff its user
and its item each appeared >= 5 times overall. There is no iteration and no
recheck of the remaining length afterwards, so a user whose original count was
>= 5 survives even if some of their interactions are dropped. Consecutive
duplicates are NOT collapsed, also matching SASRec.

Both files are Python dict literals rather than JSON, so parsing falls back to
`ast.literal_eval`.

Paper statistics: 334,728 users / 13,047 items / 1,524,022 transition edges.
"""
import ast
import collections
import datetime as dt
import gzip
import json
import os

from . import common

BASE_URL = "https://cseweb.ucsd.edu/~wckang"
GAMES_FILE = "steam_games.json.gz"
REVIEWS_FILE = "steam_reviews.json.gz"
KCORE = 5


def _parse_line(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def download_raw(raw_dir):
    return (
        common.download(f"{BASE_URL}/{GAMES_FILE}", os.path.join(raw_dir, GAMES_FILE)),
        common.download(f"{BASE_URL}/{REVIEWS_FILE}", os.path.join(raw_dir, REVIEWS_FILE)),
    )


def prepare(raw_dir, out_dir):
    games_path, reviews_path = download_raw(raw_dir)

    print("[1/4] reading games ...")
    item_text = {}
    with gzip.open(games_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                g = _parse_line(line)
            except Exception:
                continue
            gid = g.get("id")
            if gid is None:
                continue
            fields = [
                ("Title", g.get("app_name") or g.get("title", "")),
                ("Developer", g.get("developer", "")),
                ("Publisher", g.get("publisher", "")),
            ]
            for label, key in [("Genres", "genres"), ("Tags", "tags"), ("Specs", "specs")]:
                v = g.get(key)
                if isinstance(v, list) and v:
                    fields.append(
                        (label, ", ".join(x for x in (common.clean_ws(i) for i in v) if x))
                    )
            text = common.join_fields(fields)
            if text:
                item_text[str(gid)] = text
    print(f"  games with usable text: {len(item_text):,}")

    print("[2/4] reading reviews ...")
    records = []
    for line_idx, line in enumerate(gzip.open(reviews_path, "rt", encoding="utf-8")):
        try:
            r = _parse_line(line)
        except Exception:
            continue
        gid = r.get("product_id")
        if gid is None or str(gid) not in item_text:
            continue
        uid = r.get("username")
        if not uid:
            continue
        try:
            when = dt.datetime.strptime(r.get("date", ""), "%Y-%m-%d")
        except ValueError:
            continue
        records.append((uid, when, str(gid), line_idx))
    print(f"  kept reviews: {len(records):,}")

    print(f"[3/4] SASRec-style {KCORE}-core (original counts, single pass) ...")
    count_u = collections.Counter(r[0] for r in records)
    count_i = collections.Counter(r[2] for r in records)
    records = [r for r in records if count_u[r[0]] >= KCORE and count_i[r[2]] >= KCORE]
    print(f"  records after: {len(records):,}")

    # Sort by (user, date, original line) so same-day reviews keep file order.
    records.sort(key=lambda x: (x[0], x[1], x[3]))
    user_seq = collections.defaultdict(list)
    for uid, _when, gid, _li in records:
        user_seq[uid].append(gid)          # no consecutive dedup, per SASRec
    del records

    sequences = common.finalize(user_seq)
    used = {g for _, s in sequences for g in s}
    print(f"  users {len(sequences):,}  items {len(used):,}")

    print("[4/4] writing TFRecords ...")
    return common.write_dataset(out_dir, sequences, item_text)
