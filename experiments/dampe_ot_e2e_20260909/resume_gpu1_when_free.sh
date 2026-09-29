#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data1/ztx/MyModel-MDTA
PYTHON=/data1/ztx/.conda/envs/mdta/bin/python
SCRIPT=$PROJECT/experiments/dampe_ot_e2e_20260909
OUTPUT=$PROJECT/outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/ot_warmup_e2e_both
WATCH_LOG=$OUTPUT/resume_gpu1_watcher.log

mkdir -p "$OUTPUT"
exec 8>"$OUTPUT/resume_gpu1_watcher.lock"
flock -n 8 || { echo "GPU1 resume watcher already active"; exit 3; }
echo $$ > "$OUTPUT/resume_gpu1_watcher.pid"
echo "WATCH_START=$(date --iso-8601=seconds)" >> "$WATCH_LOG"

while true; do
  if pgrep -af 'train_periodic_ot.py --experiment ot_warmup_e2e_both' >/dev/null; then
    echo "TRAIN_ALREADY_ACTIVE=$(date --iso-8601=seconds)" >> "$WATCH_LOG"
    exit 0
  fi
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
  if [[ "$USED" -le 100 ]]; then
    echo "GPU1_FREE=$(date --iso-8601=seconds) USED_MIB=$USED" >> "$WATCH_LOG"
    if cd "$SCRIPT" && CUDA_VISIBLE_DEVICES=1 "$PYTHON" -B -u diagnose_sinkhorn_resume.py \
        --experiment ot_warmup_e2e_both --fold 1 --device cuda:0 \
        >> "$WATCH_LOG" 2>&1; then
      echo "CHECKPOINT_OT_DIAGNOSIS_PASSED=$(date --iso-8601=seconds)" >> "$WATCH_LOG"
      exec bash "$SCRIPT/run_fivefold.sh" ot_warmup_e2e_both 1 \
        >> "$OUTPUT/runner.log" 2>&1
    else
      RC=$?
    fi
    NOW_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
    echo "DIAGNOSIS_FAILED=$(date --iso-8601=seconds) RC=$RC USED_MIB=$NOW_USED" >> "$WATCH_LOG"
    if [[ "$NOW_USED" -le 100 ]]; then
      exit "$RC"
    fi
  fi
  sleep 60
done
