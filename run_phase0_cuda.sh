#!/usr/bin/env bash
# Phase 0 experiments, from scratch, on a CUDA machine.
#
# Usage:  bash run_phase0_cuda.sh [stage ...]
#   stages: seed1 seed2 seed3 dampower sweeps2 extended     (default: all, in this order)
#   e.g.    nohup bash run_phase0_cuda.sh seed1 > phase0_seed1.log 2>&1 &
#           nohup bash run_phase0_cuda.sh seed2 seed3 dampower > phase0_b.log 2>&1 &
# A stage is skipped if its output folder already contains random_initial_K_half.csv,
# so re-running after an interruption only redoes unfinished stages.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
SCRIPT=batched/preprint_experiments.py
COMMON="--device cuda --batch-size 1000 --n-samples 1000 --skip-after-collapse"
OUT=preprint_results_cuda

N_MAIN="50 75 85 95 100 110 125 135 145 150 160 175 185 200"
# 40000 / 60000 added so RELU/ELU at N=200 are not censored at the grid top.
MEM_MAIN="50 100 200 400 600 800 1000 2000 4000 8000 10000 20000 30000 40000 60000"

run() {  # run <out_subdir> <args...>
  local dir=$OUT/$1; shift
  if [ -f "$dir/random_initial_K_half.csv" ]; then echo "== skip $dir (done)"; return; fi
  echo "== $(date) start $dir"
  $PY $SCRIPT random-k-half $COMMON "$@" --out-dir "$dir"
  echo "== $(date) finished $dir"
}

main_seed() {
  run main_seed$1 --activations dam relu elu --n-values $N_MAIN \
    --memory-values $MEM_MAIN --dam-power 3.0 --seed $1
}

stage_seed1()    { main_seed 12345; }
stage_seed2()    { main_seed 23456; }
stage_seed3()    { main_seed 34567; }
stage_dampower() {
  for P in 2.0 4.0 5.0; do
    run dam_power${P%.0} --activations dam --n-values $N_MAIN \
      --memory-values 50 100 200 400 600 800 1000 2000 4000 8000 \
      --dam-power $P --seed 12345
  done
}
stage_sweeps2()  {
  run stable_sweeps2 --activations dam relu elu --n-values 50 100 150 200 \
    --memory-values $MEM_MAIN --stable-sweeps 2 --max-sweeps 200 --seed 12345
}
stage_extended() {
  # Most expensive; RELU/ELU will be censored for the largest N.
  run extended_N225_300 --activations dam relu elu --n-values 225 250 275 300 \
    --memory-values 2000 4000 8000 10000 20000 30000 40000 60000 80000 100000 \
    --dam-power 3.0 --seed 12345
}

STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(seed1 seed2 seed3 dampower sweeps2 extended)
for s in "${STAGES[@]}"; do "stage_$s"; done

# NOTE: half-threshold robustness (0.4 / 0.6) needs NO rerun: K_half is computed
# post-hoc from random_initial_grid.csv, so just re-derive it there.
