#!/usr/bin/env bash
set -euo pipefail

cd /data1/ztx/MyModel-MDTA
PYTHON=/data1/ztx/.conda/envs/mdta/bin/python
EXPORT=experiments/pdbbind_to_davis_transfer/scripts/cache_global_predictions.py
KERNEL=experiments/klifs85_interaction/run_residual_kernel.py
SIMILARITY=experiments/klifs85_interaction/data/similarity_audit_fold1/entity_similarities.npz

for FOLD in 2 3 4 5; do
  CHECKPOINT=outputs/Refine_experiment/davis/cold_start/drug_cold_5fold_seed42/baseline/fold_${FOLD}/best_model.pt
  CACHE=experiments/pdbbind_to_davis_transfer/data/global_predictions/fold${FOLD}.pt
  SPLIT=data/splits/davis_drug_cold_5fold_seed42/fold_${FOLD}/split.json
  OUTPUT=experiments/klifs85_interaction/outputs/residual_kernel_fold${FOLD}_v2
  EXPORT_LOG=experiments/klifs85_interaction/logs/cache_global_predictions_fold${FOLD}.log
  KERNEL_LOG=experiments/klifs85_interaction/logs/residual_kernel_fold${FOLD}_v2.log

  "$PYTHON" "$EXPORT" \
    --project /data1/ztx/MyModel-MDTA \
    --checkpoint "$CHECKPOINT" \
    --output "$CACHE" \
    --device cuda:2 \
    --batch-size 16 > "$EXPORT_LOG" 2>&1

  mkdir -p "$OUTPUT"
  "$PYTHON" "$KERNEL" \
    --global-cache "$CACHE" \
    --similarity "$SIMILARITY" \
    --split "$SPLIT" \
    --output-dir "$OUTPUT" \
    --shuffle-controls 20 > "$KERNEL_LOG" 2>&1

  echo "fold ${FOLD} complete"
done
