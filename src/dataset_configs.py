"""Dataset paths for the four release datasets (beauty, sports, toys, cds).

Each name corresponds to a folder under `<repo>/data/`. The semantic embedding
file is `t5xl.pkl` inside that folder.
"""
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_DATA_ROOT = os.path.join(_REPO_ROOT, "data")

_DATASETS = ["beauty", "sports", "toys", "cds"]

DATASET_CONFIGS = {
    name: {
        "data_dir": os.path.join(_DATA_ROOT, name),
        "emb_path": os.path.join(_DATA_ROOT, name, "t5xl.pkl"),
    }
    for name in _DATASETS
}
