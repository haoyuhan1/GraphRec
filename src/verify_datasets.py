"""Check prepared datasets against the statistics reported in the paper.

    python src/verify_datasets.py                # everything present
    python src/verify_datasets.py --dataset yelp lastfm

For each dataset this reads the testing split (whose sequences are the full
user histories) and the transition graph, then compares users / items /
interactions / average sequence length / edges against `dataset_configs.py`.
A dataset that matches on all five is a faithful reproduction. Item ids need
not match ours one-for-one, but note that a different id assignment does shift
metrics slightly: it changes how `torch.topk` breaks score ties and which items
random padding draws, which is worth about one user's worth of Recall on a
small dataset like Delicious.

Counting the big datasets means decompressing every testing shard, which takes
a few minutes for H&M and Amazon-M2. Use `--quick` to check only that the
expected files exist.
"""
import argparse
import glob
import os
import pickle

import tensorflow as tf

from dataset_configs import DATASET_CONFIGS, resolve_datasets, resolve_split_dir

tf.config.set_visible_devices([], "GPU")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def scan(data_dir):
    """Return (n_users, n_items, n_interactions).

    Items are counted from the items/ shards — that is the item ID space, which
    is what the paper reports. It can exceed the number of distinct items left
    in the sequences: LastFM and MovieLens-1M assign ids before truncating each
    user to their most recent 100 interactions, so a few items end up with an
    embedding but no surviving occurrence.
    """
    n_items = 0
    for fpath in sorted(glob.glob(os.path.join(
            resolve_split_dir(data_dir, "items"), "*.tfrecord.gz"))):
        for _ in tf.data.TFRecordDataset([fpath], compression_type="GZIP"):
            n_items += 1

    n_users = n_inter = 0
    for fpath in sorted(glob.glob(os.path.join(
            resolve_split_dir(data_dir, "testing"), "*.tfrecord.gz"))):
        for raw in tf.data.TFRecordDataset([fpath], compression_type="GZIP"):
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            n_users += 1
            n_inter += len(ex.features.feature["sequence_data"].int64_list.value)
    return n_users, n_items, n_inter


def check(name, quick=False):
    cfg = DATASET_CONFIGS[name]
    data_dir = cfg["data_dir"]
    label = f"{cfg['display']} ({name})"

    if not os.path.isdir(data_dir):
        return "MISSING", label, "not prepared"

    present = []
    for split in ("training", "evaluation", "testing", "items"):
        d = resolve_split_dir(data_dir, split, required=False)
        if not d or not glob.glob(os.path.join(d, "*.tfrecord.gz")):
            present.append(split)
    if present:
        return "INCOMPLETE", label, f"no shards in {', '.join(present)}"

    notes = []
    if not os.path.exists(cfg["emb_path"]):
        notes.append("no t5xl.pkl")
    graph_path = cfg["graph_path"]
    if quick:
        if not os.path.exists(graph_path):
            notes.append("no graph")
        return "PRESENT", label, "; ".join(notes) or "files present"

    users, items, inter = scan(data_dir)
    avg = inter / users if users else 0.0
    diffs = []
    if users != cfg["n_users"]:
        diffs.append(f"users {users:,} != {cfg['n_users']:,}")
    if items != cfg["n_items"]:
        diffs.append(f"items {items:,} != {cfg['n_items']:,}")
    if abs(avg - cfg["avg_seq_len"]) > 0.01:
        diffs.append(f"avg_len {avg:.2f} != {cfg['avg_seq_len']:.2f}")

    if os.path.exists(graph_path):
        with open(graph_path, "rb") as f:
            n_edges = len(pickle.load(f))
        if n_edges != cfg["n_edges"]:
            diffs.append(f"edges {n_edges:,} != {cfg['n_edges']:,}")
    else:
        notes.append("no graph (run prepare_data.py)")

    if diffs:
        return "MISMATCH", label, "; ".join(diffs)
    return "OK", label, "; ".join(notes) or f"{users:,} users, {items:,} items"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", nargs="+", default=["all"],
                        help="dataset name(s), or 'all' (default)")
    parser.add_argument("--quick", action="store_true",
                        help="Only check that files exist; skip the counts.")
    args = parser.parse_args()

    rows = [check(n, quick=args.quick) for n in resolve_datasets(args.dataset)]
    width = max(len(label) for _, label, _ in rows)
    print()
    for status, label, note in rows:
        print(f"  {status:<11} {label:<{width}}  {note}")

    bad = [r for r in rows if r[0] in ("MISMATCH", "INCOMPLETE")]
    print(f"\n  {sum(1 for r in rows if r[0] == 'OK')}/{len(rows)} verified")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
