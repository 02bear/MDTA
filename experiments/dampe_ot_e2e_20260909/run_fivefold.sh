#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 EXPERIMENT PHYSICAL_GPU" >&2
  exit 2
fi
EXPERIMENT=$1
GPU=$2
case "$EXPERIMENT" in
  ot_warmstart_finetune_both|ot_warmup_e2e_both) ;;
  *) echo "unknown experiment: $EXPERIMENT" >&2; exit 2 ;;
esac
PROJECT=/data1/ztx/MyModel-MDTA
PYTHON=/data1/ztx/.conda/envs/mdta/bin/python
SCRIPT=$PROJECT/experiments/dampe_ot_e2e_20260909
OUTPUT=$PROJECT/outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/$EXPERIMENT
mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/runner.lock"
flock -n 9 || { echo "runner already active" >&2; exit 3; }
cd "$PROJECT"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "EXPERIMENT=$EXPERIMENT PHYSICAL_GPU=$GPU START=$(date --iso-8601=seconds)"
for FOLD in 1 2 3 4 5; do
  DIR=$OUTPUT/fold_$FOLD
  EXTRA=()
  if [[ -f "$DIR/metrics.json" ]]; then
    echo "FOLD_$FOLD already complete"
    continue
  elif [[ -f "$DIR/latest_model.pt" ]]; then
    EXTRA+=(--resume)
  elif [[ -d "$DIR" ]] && [[ -n "$(find "$DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing ambiguous partial directory $DIR" >&2
    exit 4
  fi
  "$PYTHON" -B -u "$SCRIPT/train_periodic_ot.py" \
    --experiment "$EXPERIMENT" --fold "$FOLD" --device cuda:0 \
    --seed 42 --ot_warmup_epochs 10 --epsilon 0.001 \
    --epochs 500 --batch_size 16 --lr 0.0003 --weight_decay 0.00001 \
    --early_stop_patience 60 --early_stop_min_delta 0.0001 "${EXTRA[@]}" \
    2>&1 | tee -a "$OUTPUT/fold_${FOLD}_training.log"
  cp "$OUTPUT/fold_${FOLD}_training.log" "$DIR/training.log"
done
"$PYTHON" -B "$SCRIPT/summarize_fivefold.py" --experiment "$EXPERIMENT"
echo "FINISH=$(date --iso-8601=seconds)"
