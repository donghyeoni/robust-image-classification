#!/usr/bin/env bash
set -euo pipefail
DATA=${DATA:-data/Animals}
GPUS=(${GPUS:-0 2 3})
SEEDS=${SEEDS:-"0 1 2 3 4"}
WORKERS=${WORKERS:-16}
mkdir -p results/logs

python experiments/make_test_split.py --source "$DATA" --target data/balanced
python experiments/vectorization_benchmark.py --data-root data/balanced > results/logs/vectorization_benchmark.log 2>&1

EXPERIMENTS=${EXPERIMENTS:-"01_rgb_baseline 02_binarized_input 03_edge_preprocessing
  04_twochannel_fusion 05_wavelet_subbands 06_bitflip_robustness
  07_morphological_denoise 08_custom_pixel_denoise 09_mrf_denoise"}

jobs=()
for e in $EXPERIMENTS; do
  for s in $SEEDS; do jobs+=("$e $s"); done
done

worker() {
  local gpu=$1 k=$2 i
  for ((i = k; i < ${#jobs[@]}; i += ${#GPUS[@]})); do
    set -- ${jobs[$i]}
    CUDA_VISIBLE_DEVICES=$gpu python "experiments/$1.py" --data-root data/balanced \
      --seed "$2" --num-workers "$WORKERS" > "results/logs/$1_seed$2.log" 2>&1
  done
}
for k in "${!GPUS[@]}"; do worker "${GPUS[$k]}" "$k" & done
python experiments/pixel_restoration_study.py --data-root data/balanced \
  > results/logs/pixel_restoration_study.log 2>&1
python experiments/pipeline_figure.py --data-root data/balanced
wait
