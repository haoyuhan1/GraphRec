#!/usr/bin/env bash
# Run TGH-1 and TGH-2 over the paper's datasets.
#
#   bash scripts/run_all.sh                                  # all 14, GPU 0
#   GPU_IDS="0 1 2" bash scripts/run_all.sh                  # parallel across GPUs
#   DATASETS="beauty sports" bash scripts/run_all.sh         # subset
#
# Only datasets that are actually prepared are run; the rest are listed and
# skipped. Predictions land in predictions/<dataset>/{TGH-1,TGH-2}.pkl and logs
# in logs/<dataset>_<method>.log.
#
# --user_chunk is left unset so tgh.py picks the per-dataset default from
# dataset_configs.py (it scales as 1/n_items). Override with USER_CHUNK if your
# GPU is smaller than the ~1.5 GB score matrix those defaults assume.

set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
read -ra GPUS <<< "${GPU_IDS:-0}"
WAVE_SIZE=${#GPUS[@]}
WORKERS=${ASSEMBLE_WORKERS:-8}

ALL_DATASETS=$($PY -c "
import sys; sys.path.insert(0, 'src')
from dataset_configs import ALL_DATASETS; print(' '.join(ALL_DATASETS))
")
read -ra DATASETS <<< "${DATASETS:-$ALL_DATASETS}"

COMMON="--assemble_workers $WORKERS"
[ -n "${USER_CHUNK:-}" ] && COMMON="$COMMON --user_chunk $USER_CHUNK"

mkdir -p logs

# Keep only datasets that have been prepared.
declare -a READY=() MISSING=()
for ds in "${DATASETS[@]}"; do
    dir=$($PY -c "
import sys; sys.path.insert(0, 'src')
from dataset_configs import DATASET_CONFIGS; print(DATASET_CONFIGS['$ds']['data_dir'])
")
    if [ -d "$dir/testing" ] || [ -d "$dir/test" ]; then READY+=("$ds"); else MISSING+=("$ds"); fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Not prepared, skipping: ${MISSING[*]}"
    echo "  (run: bash scripts/prepare_all.sh)"
fi
if [ ${#READY[@]} -eq 0 ]; then
    echo "Nothing to run."; exit 1
fi

declare -a JOBS=()
for ds in "${READY[@]}"; do
    JOBS+=("$ds|TGH-1|--method TGH-1 --hop_k 7 2 1 --edge_weight_alpha 0.5")
    JOBS+=("$ds|TGH-2|--method TGH-2 --src2_hop_k 5 1 --src3_hop_k 3 1 --edge_weight_alpha 0.5")
done

echo "Datasets: ${READY[*]}"
echo "GPUs:     ${GPUS[*]}"
echo "Jobs:     ${#JOBS[@]}"

for ((i=0; i<${#JOBS[@]}; i+=WAVE_SIZE)); do
    for ((j=i; j<i+WAVE_SIZE && j<${#JOBS[@]}; j++)); do
        IFS='|' read -r ds method args <<< "${JOBS[$j]}"
        gpu=${GPUS[$((j-i))]}
        log="logs/${ds}_${method}.log"
        echo "  GPU $gpu: $ds $method"
        ( CUDA_VISIBLE_DEVICES=$gpu $PY -u src/tgh.py --dataset "$ds" \
              $args $COMMON 2>&1 | tee "$log" ) &
    done
    wait
done
echo "Done."
