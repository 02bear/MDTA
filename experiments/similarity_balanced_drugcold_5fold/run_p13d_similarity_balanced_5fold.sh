#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data1/ztx/MyModel-MDTA
PYTHON=/data1/ztx/.conda/envs/mdta/bin/python
SPLIT_ROOT=$PROJECT/data/splits/davis_drug_cold_similarity_balanced_5fold_seed42_v1
OUTPUT_ROOT=$PROJECT/outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_balanced_5fold_seed42_v1/baseline

cd "$PROJECT"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

echo "PROTOCOL=original_p13d_earlystop_only_split_changed"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PROJECT=$PROJECT"
echo "SPLIT_ROOT=$SPLIT_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "STARTED_AT=$(date --iso-8601=seconds)"
sha256sum \
  train_p13d_earlystop.py \
  datasets/davis_dataset_p13d.py \
  datasets/collate_p13d.py \
  models/model_p13d.py \
  models/drug_3d_egnn_encoder.py \
  models/protein_3d_encoder.py \
  "$SPLIT_ROOT"/fold_*/split.json

for FOLD in 1 2 3 4 5; do
  SPLIT_JSON=$SPLIT_ROOT/fold_$FOLD/split.json
  OUTPUT_DIR=$OUTPUT_ROOT/fold_$FOLD

  if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "REFUSING_TO_OVERWRITE_NONEMPTY=$OUTPUT_DIR" >&2
    exit 2
  fi
  mkdir -p "$OUTPUT_DIR"

  echo "FOLD_${FOLD}_STARTED_AT=$(date --iso-8601=seconds)"
  "$PYTHON" train_p13d_earlystop.py \
    --pairs_csv data/raw/davis/pairs.csv \
    --drug_1d_dir data/processed/davis/drug_1d_chemberta2 \
    --drug_2d_dir data/processed/davis/drug_2d \
    --drug_3d_dir data/processed/davis/drug_3d \
    --protein_1d_dir data/processed/davis/protein_1d_esm2 \
    --protein_3d_dir data/processed/davis/protein_3d_gvp \
    --split_json "$SPLIT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \
    --train_ratio 0.8 \
    --batch_size 16 \
    --num_workers 0 \
    --epochs 500 \
    --lr 0.0003 \
    --weight_decay 0.00001 \
    --early_stop_patience 60 \
    --early_stop_min_delta 0.0001 \
    --drug_1d_in_dim 768 \
    --drug_3d_node_in_dim 10 \
    --hidden_dim 128 \
    --dropout 0.1 \
    2>&1 | tee "$OUTPUT_DIR/train.log"
  echo "FOLD_${FOLD}_FINISHED_AT=$(date --iso-8601=seconds)"
done

echo "ALL_FOLDS_FINISHED_AT=$(date --iso-8601=seconds)"
