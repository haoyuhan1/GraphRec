"""Dataset registry for the 14 benchmarks used in the paper.

Each dataset lives under `<data_root>/<name>/` in the GRID TFRecord layout:

    <data_root>/<name>/
    ├── training/    *.tfrecord.gz    (user_id, sequence_data)  = seq[:-2]
    ├── evaluation/  *.tfrecord.gz                              = seq[:-1]
    ├── testing/     *.tfrecord.gz                              = seq
    ├── items/       *.tfrecord.gz    (id, text)
    ├── t5xl.pkl                      produced by generate_embeddings.py
    └── graph/transition_graph.pkl    produced by prepare_data.py

`data_root` defaults to `<repo>/data` and can be overridden with the
`GRAPHREC_DATA_ROOT` environment variable, so the data can live on a scratch
filesystem without touching the repo.

Split directories are resolved through `resolve_split_dir`, which accepts both
the `training/evaluation/testing` naming used by the prepare scripts and the
`train/validation/test` naming used by some GRID releases.

`user_chunk` is a per-dataset default for `tgh.py --user_chunk`. TGH scores a
[chunk, n_items] float32 matrix on the GPU, so the safe chunk size scales as
1/n_items; the values here target roughly 1.5 GB for that matrix. Override on
the command line if your GPU is larger or smaller.
"""
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

DATA_ROOT = os.environ.get(
    "GRAPHREC_DATA_ROOT", os.path.join(_REPO_ROOT, "data")
)

# Directory-name aliases, in the order they are probed.
_SPLIT_ALIASES = {
    "training":   ("training", "train"),
    "evaluation": ("evaluation", "validation", "valid", "val"),
    "testing":    ("testing", "test"),
    "items":      ("items",),
}


# ── Registry ──────────────────────────────────────────────────────────────────
# `prepare` names the module under src/prepare/ that builds the TFRecords;
# None means the dataset is downloaded pre-built (see README).
# Statistics are the paper's values and double as an integrity check —
# `verify_datasets.py` compares them against what is actually on disk.

_SPEC = [
    # name                 display              n_users    n_items    n_edges  avg_len  chunk  prepare
    ("beauty",             "Beauty",              22363,     12101,    114582,    8.15,  8000, None),
    ("sports",             "Sports",              35598,     18357,    180610,    7.96,  8000, None),
    ("toys",               "Toys",                19412,     11924,    102268,    7.97,  8000, None),
    ("cds",                "CDs",                 75258,     64443,    810347,   14.58,  6000, "cds"),
    ("delicious",          "Delicious",             718,      1200,      4016,    9.13,  8000, "delicious"),
    ("lastfm",             "LastFM",               1090,      3646,     30372,   34.02,  8000, "lastfm"),
    ("ml-1m",              "MovieLens-1M",         6040,      3416,    268867,   74.06,  8000, "ml1m"),
    ("yelp",               "Yelp",                30431,     20033,    219632,   10.40,  8000, "yelp"),
    ("mind",               "MIND",                48577,     39757,    824397,   28.16,  8000, "mind"),
    ("goodreads-comics",   "Goodreads-Comics",    89186,     48623,   1282693,   33.78,  8000, "goodreads"),
    ("goodreads-children", "Goodreads-Children", 163143,     55221,   1622817,   24.26,  7000, "goodreads"),
    ("steam",              "STEAM",              334728,     13047,   1524022,   12.59,  8000, "steam"),
    ("hm",                 "H&M",               1077045,    104468,  19487762,   26.01,  3500, "hm"),
    ("amazon-m2-uk",       "Amazon-M2-UK",      1182181,    494409,   1500196,    5.12,   800, "amazon_m2"),
]

DATASET_CONFIGS = {
    name: {
        "display":     display,
        "data_dir":    os.path.join(DATA_ROOT, name),
        "emb_path":    os.path.join(DATA_ROOT, name, "t5xl.pkl"),
        "graph_path":  os.path.join(DATA_ROOT, name, "graph", "transition_graph.pkl"),
        "n_users":     n_users,
        "n_items":     n_items,
        "n_edges":     n_edges,
        "avg_seq_len": avg_len,
        "user_chunk":  chunk,
        "prepare":     prepare,
    }
    for (name, display, n_users, n_items, n_edges, avg_len, chunk, prepare) in _SPEC
}

# Raw sources behind a login or licence acceptance. These cannot be fetched
# unattended: download them yourself and drop the files in
# <data_root>/_raw/<name>/. Each prepare module prints the exact filenames and
# URLs it expects. See the README for details.
MANUAL_DOWNLOAD = {"mind", "hm", "amazon-m2-uk"}

# Distributed pre-built by the GRID project rather than prepared here.
PREBUILT = {name for name, cfg in DATASET_CONFIGS.items() if cfg["prepare"] is None}

ALL_DATASETS = list(DATASET_CONFIGS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_split_dir(data_dir, split, required=True):
    """Return the on-disk directory for a logical split.

    Accepts both the `training/evaluation/testing` layout written by the prepare
    scripts and the `train/validation/test` layout of some GRID releases. Raises
    FileNotFoundError when `required` and no alias exists.
    """
    try:
        aliases = _SPLIT_ALIASES[split]
    except KeyError:
        raise ValueError(
            f"Unknown split {split!r}; expected one of {sorted(_SPLIT_ALIASES)}"
        ) from None

    for alias in aliases:
        candidate = os.path.join(data_dir, alias)
        if os.path.isdir(candidate):
            return candidate
    if required:
        raise FileNotFoundError(
            f"No {'/'.join(aliases)} directory under {data_dir}. "
            f"See the README 'Data preparation' section."
        )
    return None


def resolve_datasets(names):
    """Expand a CLI --dataset value into a list of dataset names.

    Accepts "all", a single name, or several names, and preserves order while
    dropping duplicates.
    """
    if isinstance(names, str):
        names = [names]
    out = []
    for n in names:
        if n == "all":
            out.extend(ALL_DATASETS)
        elif n in DATASET_CONFIGS:
            out.append(n)
        else:
            raise ValueError(
                f"Unknown dataset {n!r}. Available: {', '.join(ALL_DATASETS)}, all"
            )
    seen = set()
    return [n for n in out if not (n in seen or seen.add(n))]
