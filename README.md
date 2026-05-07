# GraphRec

Code for the paper *"An Embarrassingly Simple Graph Heuristic Reveals
Shortcut-Solvable Benchmarks for Sequential Recommendation"*.

We introduce **TGH** (Transition-Graph Heuristic), a training-free recommender. Two variants:

- **TGH-1** (single-source): rings 1, 2, 3 around the user's last item.
- **TGH-2** (multi-source): rings 1, 2 around the last and second-to-last items.

## Repository layout

```
GraphRec/
├── README.md
├── requirements.txt
├── src/
│   ├── tgh.py                    # TGH-1 and TGH-2 inference (single entry point)
│   ├── dataset_configs.py        # paths for the four datasets
│   ├── generate_embeddings.py    # google/flan-t5-xl item embeddings (step 2)
│   └── prepare_data.py           # build transition_graph.pkl (step 3)
├── scripts/
│   └── run_all.sh                # sweep all (dataset, method) pairs over GPUs
└── data/                         # populated per "Quick start" below
    └── <dataset>/
        ├── train/                or training/*.tfrecord.gz
        ├── validation/           or evaluation/*.tfrecord.gz
        ├── test/                 or testing/*.tfrecord.gz
        ├── items/*.tfrecord.gz   # text of all items
        ├── t5xl.pt               # produced by generate_embeddings.py
        ├── t5xl.pkl              # produced by generate_embeddings.py
        └── graph/transition_graph.pkl   # produced by prepare_data.py
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended.

## Quick start

### 1. Data Preparation

Prepare your dataset in the expected format:

```
data/<dataset>/
├── training/      # tfrecord.gz shards of training sequences
├── evaluation/    # tfrecord.gz shards of validation sequences
├── testing/       # tfrecord.gz shards of full test sequences
└── items/         # tfrecord.gz shards with item id + text
```

We use the four pre-processed Amazon datasets — **Beauty**, **Sports**, **Toys**,
and **CDs** — released alongside the [GRID project](https://github.com/snap-research/GRID/),
which in turn re-uses the splits from the [P5 paper](https://arxiv.org/abs/2203.13366).
Download the bundle from this [Google Drive link](https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view?usp=sharing)
(provided by GRID) and extract it under `data/`. After download, you should have:

```
data/beauty/{training,evaluation,testing,items}/*.tfrecord.gz
data/sports/...
data/toys/...
data/cds/...
```

(The GRID release uses folder names `train/validation/test`; both layouts
work — `tgh.py` accepts either.)

### 2. Embedding Generation with LLMs

Generate item embeddings with `google/flan-t5-xl` (last-hidden-state mean-pool
with the attention mask, max_length=128 — same recipe as GRID):

```bash
python src/generate_embeddings.py --dataset beauty
python src/generate_embeddings.py --dataset sports
python src/generate_embeddings.py --dataset toys
python src/generate_embeddings.py --dataset cds
```

This downloads `google/flan-t5-xl` from Hugging Face on first run and writes:

- `data/<dataset>/t5xl.pt` — `[n_items, 2048]` float32 tensor.
- `data/<dataset>/t5xl.pkl` — `[{"item_id": int, "embedding": list[float]}, ...]`,
  the format consumed by `tgh.py`.

Speed knobs: `--batch_size 16 --dtype bfloat16` roughly halves runtime on most
GPUs with negligible downstream impact.

### 3. Build the transition graph

```bash
python src/prepare_data.py --dataset beauty
python src/prepare_data.py --dataset sports
python src/prepare_data.py --dataset toys
python src/prepare_data.py --dataset cds
```

For each dataset, this scans `training/*.tfrecord.gz`, counts consecutive
`(a, b)` pairs in each user's sequence, and writes
`data/<dataset>/graph/transition_graph.pkl` (a `{(int, int): int}` dict). The
Beauty graph has 114,582 directed edges and builds in seconds.

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

### Sweep all four datasets and both methods

```bash
bash scripts/run_all.sh                                    # GPU 0, all 4 datasets
GPU_IDS="0 1 2 3" bash scripts/run_all.sh                  # parallel across GPUs
DATASETS="beauty sports" bash scripts/run_all.sh           # subset
```

Logs go to `logs/<dataset>_<method>.log`.

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
| `--user_chunk` | `8000` | users per GPU matmul. Lower if you OOM. |
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

Items are 0-indexed in the same item-id space as the GRID release.

