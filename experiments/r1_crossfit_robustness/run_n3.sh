#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data1/ztx/MyModel-MDTA
EXP=$PROJECT/experiments/r1_crossfit_robustness
OUT=$EXP/outputs
PYTHON=/data1/ztx/.conda/envs/mdta/bin/python
PAIRS=$PROJECT/data/raw/davis/pairs.csv

mkdir -p "$OUT/audit/splits" "$OUT/logs"

"$PYTHON" "$EXP/materialize_splits.py" --project "$PROJECT" --output "$OUT/audit/splits" \
  > "$OUT/logs/materialize_splits.log" 2>&1

"$PYTHON" "$EXP/reproduce_historical_r1.py" --project "$PROJECT" \
  --output "$OUT/audit/historical_reproduction.json" \
  > "$OUT/logs/historical_reproduction.log" 2>&1

echo "HISTORICAL_REPRODUCTION_GATE_PASSED=True"

run_one() {
  local fold=$1
  local cf=$2
  local gpu=$3
  local base="$OUT/fold_${fold}/cf_${cf}"
  mkdir -p "$base/stage_a_epoch_selection" "$base/stage_b_strict_refit"
  if [[ ! -f "$base/stage_a_epoch_selection/result.json" ]]; then
    echo "START Stage-A fold=$fold cf=$cf physical_gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EXP/train_crossfit_p13d.py" \
      --project "$PROJECT" --output-root "$OUT" --pairs-csv "$PAIRS" \
      --fold "$fold" --cf "$cf" --phase stage-a --device cuda:0 \
      > "$base/stage_a_epoch_selection/train.log" 2>&1
  fi
  if [[ ! -f "$base/stage_b_strict_refit/result.json" ]]; then
    echo "START Stage-B fold=$fold cf=$cf physical_gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EXP/train_crossfit_p13d.py" \
      --project "$PROJECT" --output-root "$OUT" --pairs-csv "$PAIRS" \
      --fold "$fold" --cf "$cf" --phase stage-b --device cuda:0 \
      > "$base/stage_b_strict_refit/train.log" 2>&1
  fi
  echo "COMPLETE fold=$fold cf=$cf physical_gpu=$gpu"
}

fold1_worker_zero() {
  run_one 1 1 0
  run_one 1 3 0
  run_one 1 5 0
}

fold1_worker_one() {
  run_one 1 2 1
  run_one 1 4 1
}

remaining_worker_zero() {
  run_one 2 2 0
  run_one 2 4 0
  run_one 3 1 0
  run_one 3 3 0
  run_one 3 5 0
  run_one 4 2 0
  run_one 4 4 0
  run_one 5 1 0
  run_one 5 3 0
  run_one 5 5 0
}

remaining_worker_one() {
  run_one 2 1 1
  run_one 2 3 1
  run_one 2 5 1
  run_one 3 2 1
  run_one 3 4 1
  run_one 4 1 1
  run_one 4 3 1
  run_one 4 5 1
  run_one 5 2 1
  run_one 5 4 1
}

fold1_worker_zero &
PID_ZERO=$!
fold1_worker_one &
PID_ONE=$!
FAILED=0
wait "$PID_ZERO" || FAILED=1
wait "$PID_ONE" || FAILED=1
if [[ "$FAILED" -ne 0 ]]; then
  echo "N3_TRAINING_STOPPED_DUE_TO_RUN_FAILURE=True"
  exit 2
fi

"$PYTHON" "$EXP/audit_oof_fold.py" --project "$PROJECT" --output-root "$OUT" --fold 1 \
  > "$OUT/logs/fold_1_oof_technical_gate.log" 2>&1
echo "FOLD1_OOF_TECHNICAL_GATE_PASSED=True"

remaining_worker_zero &
PID_ZERO=$!
remaining_worker_one &
PID_ONE=$!
FAILED=0
wait "$PID_ZERO" || FAILED=1
wait "$PID_ONE" || FAILED=1
if [[ "$FAILED" -ne 0 ]]; then
  echo "N3_TRAINING_STOPPED_DUE_TO_RUN_FAILURE=True"
  exit 2
fi

for FOLD in 2 3 4 5; do
  "$PYTHON" "$EXP/audit_oof_fold.py" --project "$PROJECT" --output-root "$OUT" --fold "$FOLD" \
    > "$OUT/logs/fold_${FOLD}_oof_technical_gate.log" 2>&1
done
echo "ALL_OOF_TECHNICAL_GATES_PASSED=True"

"$PYTHON" "$EXP/evaluate_n3.py" --project "$PROJECT" --output-root "$OUT" \
  > "$OUT/logs/evaluate_n3.log" 2>&1
echo "N3_ALL_COMPLETE=True"
