"""Build the item-transition graph for one or more prepared datasets.

Counts consecutive `(a, b)` pairs in every training sequence and writes
`<data_dir>/graph/transition_graph.pkl` as a `{(int, int): int}` dict.

    python src/prepare_data.py --dataset beauty
    python src/prepare_data.py --dataset all

Run this after the TFRecords exist (see `python -m src.prepare`) and before
`tgh.py`. The edge count is printed and compared against the paper's value.
"""
import argparse
import collections
import glob
import os
import pickle

import tensorflow as tf

from dataset_configs import DATASET_CONFIGS, resolve_datasets, resolve_split_dir

tf.config.set_visible_devices([], "GPU")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def build_transition_graph(data_dir, overwrite=False):
    out_path = os.path.join(data_dir, "graph", "transition_graph.pkl")
    if os.path.exists(out_path) and not overwrite:
        with open(out_path, "rb") as f:
            edge_counts = pickle.load(f)
        print(f"[skip] {out_path} exists ({len(edge_counts):,} edges); "
              f"pass --overwrite to rebuild.")
        return edge_counts

    train_dir = resolve_split_dir(data_dir, "training")
    paths = sorted(glob.glob(os.path.join(train_dir, "*.tfrecord.gz")))
    if not paths:
        raise FileNotFoundError(f"No *.tfrecord.gz under {train_dir}.")

    print(f"Building transition graph from {len(paths)} shards ...")
    edge_counts = collections.Counter()
    for fpath in paths:
        for raw in tf.data.TFRecordDataset([fpath], compression_type="GZIP"):
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            seq = list(ex.features.feature["sequence_data"].int64_list.value)
            for a, b in zip(seq[:-1], seq[1:]):
                edge_counts[(int(a), int(b))] += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(dict(edge_counts), f)
    print(f"  {len(edge_counts):,} edges  ->  {out_path}")
    return edge_counts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", nargs="+", default=["beauty"],
                        help="dataset name(s), or 'all'")
    parser.add_argument("--data_dir", default=None,
                        help="Override the data dir (single dataset only).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    names = resolve_datasets(args.dataset)
    if args.data_dir and len(names) > 1:
        parser.error("--data_dir only makes sense with a single dataset")

    missing = []
    for name in names:
        cfg = DATASET_CONFIGS[name]
        data_dir = args.data_dir or cfg["data_dir"]
        if not os.path.isdir(data_dir):
            print(f"[skip] {name}: {data_dir} not found.")
            missing.append(name)
            continue
        print(f"\n=== {cfg['display']} ({name}) ===")
        edge_counts = build_transition_graph(data_dir, overwrite=args.overwrite)
        expected = cfg["n_edges"]
        if len(edge_counts) != expected:
            print(f"  WARNING: {len(edge_counts):,} edges, paper reports {expected:,}.")

    if missing:
        print(f"\nNot prepared yet: {', '.join(missing)}")
        print("Run `python -m src.prepare <name>` first (see the README).")


if __name__ == "__main__":
    main()
