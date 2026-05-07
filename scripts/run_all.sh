#!/usr/bin/env bash
# Run TGH-1 and TGH-2 over the four release datasets.
#
# Usage:
#   bash scripts/run_all.sh                  # default GPUs / workers
#   GPU_IDS="0 1" DATASETS="beauty sports" bash scripts/run_all.sh
#
# Predictions land in predictions/<dataset>/{TGH-1,TGH-2}.pkl.

set -e
cd "$(dirname "$0")/.."

read -ra GPUS <<< "${GPU_IDS:-0}"
read -ra DATASETS <<< "${DATASETS:-beauty sports toys cds}"
WORKERS=${ASSEMBLE_WORKERS:-8}
USER_CHUNK=${USER_CHUNK:-4000}

PY=${PYTHON:-python}
WAVE_SIZE=${#GPUS[@]}

mkdir -p logs

# Build the (dataset, method) job list.
declare -a JOBS=()
for ds in "${DATASETS[@]}"; do
    JOBS+=("$ds|TGH-1|--method TGH-1 --hop_k 7 2 1 --edge_weight_alpha 0.5")
    JOBS+=("$ds|TGH-2|--method TGH-2 --src2_hop_k 5 1 --src3_hop_k 3 1 --edge_weight_alpha 0.5")
done

COMMON="--user_chunk $USER_CHUNK --assemble_workers $WORKERS"

echo "Datasets: ${DATASETS[*]}"
echo "GPUs:     ${GPUS[*]}"
echo "Total jobs: ${#JOBS[@]}"

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
