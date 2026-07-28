# GraphRec

Code for the paper *"An Embarrassingly Simple Graph Heuristic Reveals
Shortcut-Solvable Benchmarks for Sequential Recommendation"*.

We introduce **TGH** (Transition-Graph Heuristic), a training-free recommender. Two variants:

- **TGH-1** (single-source): rings 1, 2, 3 around the user's last item.
- **TGH-2** (multi-source): rings 1, 2 around the last and second-to-last items.

## Quick start

```bash
# 1. Prepare a dataset
python -m src.prepare yelp

# 2. Generate embeddings and transition graph
python src/generate_embeddings.py --dataset yelp
python src/prepare_data.py --dataset yelp

# 3. Verify data statistics
python src/verify_datasets.py --dataset yelp

# 4. Reproduce TGH results
DATASETS="yelp" bash scripts/run_all.sh
```

See the [detailed walkthrough](#detailed-walkthrough) below for each step and
the other 13 datasets.

## Repository layout

```
GraphRec/
├── README.md
├── requirements.txt
├── src/
│   ├── tgh.py                    # TGH-1 and TGH-2 inference (single entry point)
│   ├── dataset_configs.py        # the 14 datasets: paths, statistics, defaults
│   ├── prepare/                  # raw download -> TFRecords, one module per source
│   │   ├── common.py             # shared remap / split / sharding / cleaning
│   │   └── cds.py, yelp.py, ...
│   ├── generate_embeddings.py    # google/flan-t5-xl item embeddings
│   ├── prepare_data.py           # build transition_graph.pkl
│   └── verify_datasets.py        # check prepared data against the paper's stats
├── scripts/
│   ├── prepare_all.sh            # one-shot: download -> TFRecords -> embeddings -> graph
│   └── run_all.sh                # run TGH-1 and TGH-2 over all datasets on GPUs
└── data/                         # populated per "Data preparation" below
    ├── _raw/<dataset>/           # cached raw downloads
    └── <dataset>/
        ├── training/             or train/*.tfrecord.gz
        ├── evaluation/           or validation/*.tfrecord.gz
        ├── testing/              or test/*.tfrecord.gz
        ├── items/*.tfrecord.gz   # text of all items
        ├── t5xl.pt               # produced by generate_embeddings.py
        ├── t5xl.pkl              # produced by generate_embeddings.py
        └── graph/transition_graph.pkl   # produced by prepare_data.py
```

Set `GRAPHREC_DATA_ROOT` to keep the data somewhere other than `./data`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended.

## Datasets

The paper evaluates on 14 sequential-recommendation benchmarks. We do not
redistribute any of them — several forbid it — so this repository ships the
download-and-preprocess code instead. Every dataset is rebuilt from its
original public source, and `verify_datasets.py` checks the result against the
statistics below.

| Dataset | #Users | #Items | #Edges | Avg. len | Source | Auto |
|---|---:|---:|---:|---:|---|:--:|
| Beauty | 22,363 | 12,101 | 114,582 | 8.15 | GRID bundle | — |
| Sports | 35,598 | 18,357 | 180,610 | 7.96 | GRID bundle | — |
| Toys | 19,412 | 11,924 | 102,268 | 7.97 | GRID bundle | — |
| CDs | 75,258 | 64,443 | 810,347 | 14.58 | Amazon Reviews 2014 (SNAP) | ✅ |
| Delicious | 718 | 1,200 | 4,016 | 9.13 | HetRec 2011 Delicious-2K | ✅ |
| LastFM | 1,090 | 3,646 | 30,372 | 34.02 | HetRec 2011 Last.fm-2K | ✅ |
| MovieLens-1M | 6,040 | 3,416 | 268,867 | 74.06 | GroupLens ml-1m | ✅ |
| Yelp | 30,431 | 20,033 | 219,632 | 10.40 | LETTER's Yelp benchmark | ✅ |
| MIND | 48,577 | 39,757 | 824,397 | 28.16 | MIND-small | 🔒 |
| Goodreads-Comics | 89,186 | 48,623 | 1,282,693 | 33.78 | UCSD Goodreads | ✅ |
| Goodreads-Children | 163,143 | 55,221 | 1,622,817 | 24.26 | UCSD Goodreads | ✅ |
| STEAM | 334,728 | 13,047 | 1,524,022 | 12.59 | UCSD Steam | ✅ |
| H&M | 1,077,045 | 104,468 | 19,487,762 | 26.01 | Kaggle H&M | 🔒 |
| Amazon-M2-UK | 1,182,181 | 494,409 | 1,500,196 | 5.12 | KDD Cup 2023 Amazon-M2 | 🔒 |

✅ downloaded automatically 🔒 you must download the raw files yourself
(licence acceptance or competition login) — see below

All 14 use the same leave-2-out split: `training = seq[:-2]`,
`evaluation = seq[:-1]`, `testing = seq` (target `seq[-1]`). Per-dataset
filtering protocols differ and are documented at the top of each module in
[`src/prepare/`](src/prepare/) — they are not interchangeable, and the exact
choices are what reproduce the counts above.

## Detailed walkthrough

### 1. Data preparation

Everything at once:

```bash
bash scripts/prepare_all.sh
```

That runs raw download → TFRecords → item embeddings → transition graph for
every dataset it can fetch unattended, then verifies each against the table
above. Or per dataset:

```bash
python -m src.prepare yelp          # raw -> data/yelp/{training,evaluation,testing,items}
python src/generate_embeddings.py --dataset yelp
python src/prepare_data.py --dataset yelp
python src/verify_datasets.py --dataset yelp
```

`--dataset all` works for each of these. Raw downloads are cached under
`data/_raw/<dataset>/`, so re-running is cheap.

**Beauty / Sports / Toys** are the pre-processed splits released with the
[GRID project](https://github.com/snap-research/GRID/), which re-uses the
[P5 paper](https://arxiv.org/abs/2203.13366)'s splits. We link them rather than
rebuilding so the item-id space stays identical to prior work. Download the
bundle from GRID's [Google Drive link](https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view?usp=sharing)
and extract it under `data/`, giving
`data/beauty/{training,evaluation,testing,items}/*.tfrecord.gz`. The GRID
release names its folders `train/validation/test`; both layouts work.

**MIND**, **H&M** and **Amazon-M2-UK** must be downloaded by hand — they sit
behind a licence acceptance or a competition login and cannot be scripted.
Fetch the raw files and drop them in `data/_raw/<dataset>/`:

| Dataset | Files | Where |
|---|---|---|
| MIND | `MINDsmall_train.zip`, `MINDsmall_dev.zip` | [HF mirror](https://huggingface.co/datasets/yjw1029/MIND) linked from [msnews.github.io](https://msnews.github.io/); sign in and accept the MSR licence |
| H&M | `articles.csv`, `transactions_train.csv` | [Kaggle competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) |
| Amazon-M2-UK | `products_train.csv`, `sessions_train.csv` | [KDD Cup 2023](https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge) |

Running `python -m src.prepare <name>` without them prints exactly this list
with the path it expects, then stops. `scripts/prepare_all.sh` skips all three
rather than failing. For MIND only, setting `HF_TOKEN` to a read token from an
account that has accepted the licence lets the script fetch the archives for
you.

### 2. Item embeddings

`generate_embeddings.py` encodes each item's text with `google/flan-t5-xl`
(last-hidden-state mean-pool with the attention mask, max_length=128 — the GRID
recipe) and writes:

- `data/<dataset>/t5xl.pt` — `[n_items, 2048]` float32 tensor.
- `data/<dataset>/t5xl.pkl` — `[{"item_id": int, "embedding": list[float]}, ...]`,
  the format consumed by `tgh.py`.

The ~10 GB encoder is downloaded on first run and loaded once per invocation,
so `--dataset all` is much faster than 14 separate calls. Speed knobs:
`--batch_size 16 --dtype bfloat16` roughly halves runtime with negligible
downstream impact.

### 3. Transition graph

`prepare_data.py` scans `training/*.tfrecord.gz`, counts consecutive `(a, b)`
pairs, and writes `data/<dataset>/graph/transition_graph.pkl` (a
`{(int, int): int}` dict). It prints the edge count and flags any disagreement
with the table above. Beauty's 114,582 edges build in seconds; H&M's 19.5M take
a few minutes.

### 4. Run TGH-1 / TGH-2

```bash
# TGH-1 (single-source, rings 1/2/3 around the last item)
python src/tgh.py --dataset beauty --method TGH-1 \
    --hop_k 7 2 1 --edge_weight_alpha 0.5

# TGH-2 (multi-source, rings 1/2 around the last and second-to-last items)
python src/tgh.py --dataset beauty --method TGH-2 \
    --src2_hop_k 5 1 --src3_hop_k 3 1 --edge_weight_alpha 0.5
```

The script prints Recall@{1,5,10} and NDCG@{1,5,10} on the test split and
writes top-10 predictions to `predictions/<dataset>/<method>.pkl` as a list of
`{"user_id", "item_ids", "gt_ids"}` records.

### Sweep all datasets and both methods

```bash
bash scripts/run_all.sh                                    # GPU 0, all 14 datasets
GPU_IDS="0 1 2" bash scripts/run_all.sh                    # parallel across GPUs
DATASETS="beauty sports" bash scripts/run_all.sh           # subset
```

Datasets that have not been prepared are listed and skipped rather than
failing. Logs go to `logs/<dataset>_<method>.log`.

## Key arguments

| flag | default | description |
|------|---------|-------------|
| `--method` | `TGH-1` | `TGH-1` or `TGH-2`. |
| `--hop_k` | `7 2 1` | TGH-1 per-ring budget (sum = top-K returned). |
| `--src2_hop_k` | `5 1` | TGH-2 source-2 (last item) per-ring budget. |
| `--src3_hop_k` | `3 1` | TGH-2 source-3 (`history[-2]`) per-ring budget. |
| `--edge_weight_alpha` | `0.5` | weight on `log1p(edge_count)/row_max` added to ring-1 scores. |
| `--last_n` | `1` | for TGH-1, average the embeddings of the last `n` items into the anchor. |
| `--max_hops` | `2` | max ring depth materialised globally. |
| `--max_explicit_depth` | `None` | cap on rings stored explicitly (deeper rings become catch-all `NOT(visited ∪ history)`). |
| `--semantic_pad` | off | when ring picks under-fill, pad with anchor-similar items from the catch-all pool instead of random sampling. |
| `--user_chunk` | per-dataset | users per GPU matmul. Defaults come from `dataset_configs.py` and scale as `1/n_items` (8000 for Beauty, 800 for Amazon-M2-UK). Lower if you OOM. |
| `--assemble_workers` | `8` | CPU workers for the per-user overflow assembly fallback (only used by users whose ring picks don't fully fill `hop_k`). |

## Output format

```python
import pickle
preds = pickle.load(open("predictions/beauty/TGH-1.pkl", "rb"))
# [
#   {"user_id": 0, "item_ids": [1234, 567, ...], "gt_ids": [target_item_id]},
#   ...
# ]
```

Items are 0-indexed. For Beauty/Sports/Toys the id space is the GRID release's;
for the other 11 it is assigned by `src/prepare/`.

A rebuild that matches the dataset table reproduces the paper's numbers, but not
always to the last decimal. Item ids need not match ours one-for-one, and a
different assignment changes how `torch.topk` breaks score ties and which items
`--seed` draws for random padding. On small datasets where the rings under-fill
for most users — Delicious under-fills for 649 of 718 — that is worth about one
user's worth of Recall, the same spread you get from changing `--seed` alone.

## Surveyed papers

The full list of **94** generative-recommendation papers surveyed in
this work is available in [`SURVEYED_PAPERS.md`](SURVEYED_PAPERS.md).
