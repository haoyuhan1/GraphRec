#!/usr/bin/env bash
# One-shot data preparation: raw download -> TFRecords -> item embeddings ->
# transition graph, for every dataset in the paper.
#
#   bash scripts/prepare_all.sh                      # everything it can do unattended
#   DATASETS="yelp lastfm" bash scripts/prepare_all.sh
#   STAGES="prepare graph" bash scripts/prepare_all.sh   # skip the embedding step
#
# Environment:
#   GRAPHREC_DATA_ROOT  where data lives (default: <repo>/data)
#   DATASETS            space-separated names, or "all" (default)
#   STAGES              subset of "prepare embed graph" (default: all three)
#   HF_TOKEN            required for MIND; see README
#   EMBED_ARGS          extra flags for generate_embeddings.py
#                       (e.g. "--dtype bfloat16 --batch_size 16")
#
# Beauty / Sports / Toys are downloaded pre-built from the GRID project — see
# the README. H&M and Amazon-M2 need their raw CSVs placed by hand; this script
# skips them with a note rather than failing.

set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
DATASETS=${DATASETS:-all}
STAGES=${STAGES:-"prepare embed graph"}
EMBED_ARGS=${EMBED_ARGS:-}

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

echo "data root : ${GRAPHREC_DATA_ROOT:-$(pwd)/data}"
echo "datasets  : $DATASETS"
echo "stages    : $STAGES"
echo

rc=0

if has_stage prepare; then
    echo "=== 1. raw -> TFRecords ==========================================="
    # --keep_going so one unavailable source doesn't abort the rest;
    # --skip_manual so the Kaggle-gated pair is reported, not fatal.
    $PY -m src.prepare $DATASETS --keep_going --skip_manual || rc=1
fi

if has_stage embed; then
    echo
    echo "=== 2. item embeddings (google/flan-t5-xl) ========================"
    # The encoder is loaded once and reused across datasets.
    $PY src/generate_embeddings.py --dataset $DATASETS $EMBED_ARGS || rc=1
fi

if has_stage graph; then
    echo
    echo "=== 3. transition graphs =========================================="
    $PY src/prepare_data.py --dataset $DATASETS || rc=1
fi

echo
echo "=== verification ==================================================="
$PY src/verify_datasets.py --dataset $DATASETS || rc=1

echo
if [ $rc -ne 0 ]; then
    echo "Some datasets are not ready. Common reasons:"
    echo "  beauty/sports/toys : download the GRID bundle (see README)"
    echo "  mind               : needs HF_TOKEN"
    echo "  hm, amazon-m2-uk   : place the Kaggle CSVs under <data_root>/_raw/<name>/"
fi
exit $rc
