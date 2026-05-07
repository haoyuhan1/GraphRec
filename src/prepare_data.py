"""Build the item-transition graph for a GRID-format dataset.

Given `data/<dataset>/training/*.tfrecord.gz`, count consecutive (a, b) pairs
in each user's training sequence and save the result to
`data/<dataset>/graph/transition_graph.pkl` as a `{(int, int): int}` dict.

Run once per dataset (after the GRID data is in place):
    python src/prepare_data.py --dataset beauty
    python src/prepare_data.py --dataset sports
    python src/prepare_data.py --dataset toys
"""
import argparse
import collections
import glob
import os
import pickle

import tensorflow as tf

from dataset_configs import DATASET_CONFIGS

tf.config.set_visible_devices([], "GPU")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def build_transition_graph(data_dir):
    out_path = os.path.join(data_dir, "graph", "transition_graph.pkl")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists.")
        return
    paths = sorted(glob.glob(os.path.join(data_dir, "training", "*.tfrecord.gz")))
    if not paths:
        raise FileNotFoundError(
            f"No training/*.tfrecord.gz under {data_dir}. Place the GRID-format "
            f"training shards there first."
        )
    print(f"Building transition graph from {len(paths)} shards ...")
    edge_counts = collections.Counter()
    for fpath in paths:
        ds = tf.data.TFRecordDataset([fpath], compression_type="GZIP")
        for raw in ds:
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            seq = list(ex.features.feature["sequence_data"].int64_list.value)
            for a, b in zip(seq[:-1], seq[1:]):
                edge_counts[(int(a), int(b))] += 1
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(dict(edge_counts), f)
    print(f"  {len(edge_counts):,} edges  →  {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="beauty",
                        choices=list(DATASET_CONFIGS))
    parser.add_argument("--data_dir", default=None,
                        help="Override data dir (default: data/<dataset>).")
    args = parser.parse_args()
    data_dir = args.data_dir or DATASET_CONFIGS[args.dataset]["data_dir"]
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"{data_dir} not found. Place the GRID-format dataset there first."
        )
    build_transition_graph(data_dir)


if __name__ == "__main__":
    main()
