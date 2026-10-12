#!/usr/bin/env bash
# Phase 0 experiments, from scratch, on a CUDA machine.
# Usage:  bash run_phase0_cuda.sh            (run everything, sequentially)
#         nohup bash run_phase0_cuda.sh > phase0.log 2>&1 &    (detached)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
SCRIPT=batched/preprint_experiments.py
COMMON="--device cuda --batch-size 1000 --n-samples 1000 --skip-after-collapse"
OUT=preprint_results_cuda

N_MAIN="50 75 85 95 100 110 125 135 145 150 160 175 185 200"
# 40000 / 60000 added so RELU/ELU at N=200 are not censored at the grid top.
MEM_MAIN="50 100 200 400 600 800 1000 2000 4000 8000 10000 20000 30000 40000 60000"

# 1. main grid, 3 seeds (error bars on K_half)
for SEED in 12345 23456 34567; do
  $PY $SCRIPT random-k-half $COMMON --activations dam relu elu \
    --n-values $N_MAIN --memory-values $MEM_MAIN \
    --dam-power 3.0 --seed $SEED --out-dir $OUT/main_seed$SEED
done

# 2. dam_power sensitivity (DAM only, cheap)
for P in 2.0 4.0 5.0; do
  $PY $SCRIPT random-k-half $COMMON --activations dam \
    --n-values $N_MAIN --memory-values 50 100 200 400 600 800 1000 2000 4000 8000 \
    --dam-power $P --seed 12345 --out-dir $OUT/dam_power${P%.0}
done

# 3. convergence-criterion robustness (stable-sweeps 2, subset of N)
$PY $SCRIPT random-k-half $COMMON --activations dam relu elu \
  --n-values 50 100 150 200 --memory-values $MEM_MAIN \
  --stable-sweeps 2 --max-sweeps 200 --seed 12345 --out-dir $OUT/stable_sweeps2

# 4. extended N (most expensive; RELU/ELU will be censored for the largest N)
$PY $SCRIPT random-k-half $COMMON --activations dam relu elu \
  --n-values 225 250 275 300 \
  --memory-values 2000 4000 8000 10000 20000 30000 40000 60000 80000 100000 \
  --dam-power 3.0 --seed 12345 --out-dir $OUT/extended_N225_300

# NOTE: half-threshold robustness (0.4 / 0.6) needs NO rerun: K_half is
# computed post-hoc from random_initial_grid.csv, so just re-derive it there.
