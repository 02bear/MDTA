#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data1/ztx/MyModel-MDTA
SCRIPT=$PROJECT/experiments/dampe_ot_e2e_20260909
ROOT=$PROJECT/outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final
A=ot_warmstart_finetune_both
B=ot_warmup_e2e_both

mapfile -t USED < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
for GPU in 0 1; do
  if (( ${USED[$GPU]} > 100 )); then
    echo "GPU $GPU is no longer free: ${USED[$GPU]} MiB" >&2
    exit 3
  fi
done
for EXPERIMENT in "$A" "$B"; do
  DIR=$ROOT/$EXPERIMENT
  if [[ -e "$DIR/runner.pid" ]] || [[ -e "$DIR/fivefold_summary.json" ]]; then
    echo "Refusing duplicate launch: $DIR" >&2
    exit 4
  fi
  mkdir -p "$DIR"
done

nohup bash "$SCRIPT/run_fivefold.sh" "$A" 0 > "$ROOT/$A/runner.log" 2>&1 < /dev/null &
PID_A=$!
echo "$PID_A" > "$ROOT/$A/runner.pid"
nohup bash "$SCRIPT/run_fivefold.sh" "$B" 1 > "$ROOT/$B/runner.log" 2>&1 < /dev/null &
PID_B=$!
echo "$PID_B" > "$ROOT/$B/runner.pid"
printf 'A_PID=%s GPU=0\nB_PID=%s GPU=1\n' "$PID_A" "$PID_B"
