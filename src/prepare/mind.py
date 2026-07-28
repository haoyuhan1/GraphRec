"""MIND-small → GRID TFRecords.

Source: the Microsoft News Dataset (small). YOU MUST DOWNLOAD THIS YOURSELF —
Microsoft's original Azure endpoints have been retired and msnews.github.io now
points at a Hugging Face mirror whose repository is gated behind the MSR
license, so it cannot be fetched unattended. Put these two archives in the raw
directory:

    MINDsmall_train.zip   news.tsv (titles only; behaviors unused)
    MINDsmall_dev.zip     news.tsv + behaviors.tsv

As a convenience, if `HF_TOKEN` holds a Hugging Face token for an account that
has accepted the license, this module will fetch them for you.

Sequence construction:
  - dev split only (its ClickHist covers weeks 1-5, label=1 impressions week 6).
  - per user: the latest ClickHist, then the label=1 clicks in impression-time
    order, deduplicated, keeping first occurrence.
  - item text = news title only, matching the MIND paper's fair-comparison setup.
  - news.tsv from train and dev are unioned purely for title coverage.

Paper statistics: 48,577 users / 39,757 items / 824,397 transition edges.
"""
import collections
import csv
import datetime as dt
import os
import sys

from . import common

HF_REPO = "https://huggingface.co/datasets/yjw1029/MIND/resolve/main"
ARCHIVES = ["MINDsmall_train.zip", "MINDsmall_dev.zip"]

_MANUAL = """\
Download the two archives from the mirror linked by https://msnews.github.io/ :

    https://huggingface.co/datasets/yjw1029/MIND

Sign in with a (free) Hugging Face account and accept the MSR license, then
either download MINDsmall_train.zip and MINDsmall_dev.zip by hand into the raw
directory above, or create a read token at
https://huggingface.co/settings/tokens and let this script fetch them:

    HF_TOKEN=hf_xxx python -m src.prepare mind\
"""


def _download_gated(url, dest):
    """Download a gated Hugging Face file using HF_TOKEN."""
    import urllib.request

    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  [cached] {dest}")
        return dest
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    tmp = dest + ".part"
    tty = sys.stdout.isatty()
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as out:
        total = next_mark = 0
        while chunk := r.read(1 << 20):
            out.write(chunk)
            total += len(chunk)
            mb = total / 1024 / 1024
            if tty:
                sys.stdout.write(f"\r    {mb:.1f} MB")
                sys.stdout.flush()
            elif mb >= next_mark:
                print(f"    {mb:.1f} MB", flush=True)
                next_mark += 50
    if tty:
        sys.stdout.write("\n")
    os.replace(tmp, dest)
    return dest


def download_raw(raw_dir):
    """Fetch and unpack both archives. Returns (train_dir, dev_dir)."""
    dirs = []
    for archive in ARCHIVES:
        stem = archive[:-4]
        dest_dir = os.path.join(raw_dir, stem)
        if os.path.exists(os.path.join(dest_dir, "news.tsv")):
            print(f"  [cached] {dest_dir}")
            dirs.append(dest_dir)
            continue
        archive_path = os.path.join(raw_dir, archive)
        if not os.path.exists(archive_path):
            _download_gated(f"{HF_REPO}/{archive}", archive_path)
        common.require_manual(raw_dir, [archive], "MIND", _MANUAL)
        common.unzip(archive_path, dest_dir)
        dirs.append(dest_dir)
    return dirs


def prepare(raw_dir, out_dir):
    train_dir, dev_dir = download_raw(raw_dir)
    csv.field_size_limit(sys.maxsize)

    print("[1/3] reading news titles (train + dev) ...")
    title = {}
    for d in (train_dir, dev_dir):
        with open(os.path.join(d, "news.tsv"), encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                nid, headline = row[0], row[3]
                if headline and nid not in title:
                    title[nid] = headline
    print(f"  {len(title):,} news with a title")

    print("[2/3] building per-user sequences (dev split) ...")
    latest_hist = {}                              # uid -> (time, [nid, ...])
    clicks = collections.defaultdict(list)        # uid -> [(time, nid), ...]
    with open(os.path.join(dev_dir, "behaviors.tsv"), encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            _, uid, time_str, hist, imps = row
            when = dt.datetime.strptime(time_str, "%m/%d/%Y %I:%M:%S %p")
            prev = latest_hist.get(uid)
            if prev is None or when > prev[0]:
                latest_hist[uid] = (when, hist.split() if hist.strip() else [])
            for token in imps.split():
                nid, label = token.rsplit("-", 1)
                if label == "1":
                    clicks[uid].append((when, nid))

    user_seq = {}
    for uid in set(latest_hist) | set(clicks):
        seen, seq = set(), []
        for nid in latest_hist.get(uid, (None, []))[1]:
            if nid in title and nid not in seen:
                seen.add(nid)
                seq.append(nid)
        for _when, nid in sorted(clicks.get(uid, []), key=lambda x: x[0]):
            if nid in title and nid not in seen:
                seen.add(nid)
                seq.append(nid)
        user_seq[uid] = seq

    sequences = common.finalize(user_seq)
    used = {n for _, s in sequences for n in s}
    print(f"  users {len(sequences):,}  items {len(used):,}")

    print("[3/3] writing TFRecords ...")
    return common.write_dataset(out_dir, sequences, title)
