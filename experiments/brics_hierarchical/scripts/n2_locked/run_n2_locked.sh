#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data1/ztx/MyModel-MDTA
PYTHON=/data1/ztx/.conda/envs/mdta/bin/python
SCRIPTS=$PROJECT/experiments/brics_hierarchical/scripts/n2_locked
OUT=$PROJECT/experiments/brics_hierarchical/outputs/brics_r1_complementarity_5fold
PAIRS=$PROJECT/data/raw/davis/pairs.csv
MAPPING=$PROJECT/experiments/brics_hierarchical/data/brics_mappings.pt
BASE_SCRIPTS=$PROJECT/experiments/brics_hierarchical/scripts
mkdir -p "$OUT/logs"

echo "STAGE=PY_COMPILE"
"$PYTHON" -m py_compile \
  "$SCRIPTS/n2_build_inner_splits.py" \
  "$SCRIPTS/n2_cache_label_free.py" \
  "$SCRIPTS/n2_nested_train.py" \
  "$SCRIPTS/n2_evaluate_locked.py" \
  "$SCRIPTS/n2_augment_brics_chemistry.py"

echo "STAGE=INNER_SPLITS"
"$PYTHON" "$SCRIPTS/n2_build_inner_splits.py" \
  --split-root "$PROJECT/data/splits/davis_drug_cold_5fold_seed42" \
  --output-root "$OUT/splits"

echo "STAGE=FOLD1_REPRODUCTION_PROVENANCE"
"$PYTHON" "$BASE_SCRIPTS/eval_brics_r1_complementarity.py" \
  --project "$PROJECT" \
  --global-cache "$PROJECT/experiments/pdbbind_to_davis_transfer/data/global_predictions/fold1_with_features.pt" \
  --drug-cache "$PROJECT/experiments/brics_hierarchical/data/fold1_drug_encoder_inputs_chem.pt" \
  --similarity "$PROJECT/experiments/klifs85_interaction/data/similarity_audit_fold1/entity_similarities.npz" \
  --split "$PROJECT/data/splits/davis_drug_cold_5fold_seed42/fold_1/split.json" \
  --p13d-checkpoint "$PROJECT/outputs/Refine_experiment/davis/cold_start/drug_cold_5fold_seed42/baseline/fold_1/best_model.pt" \
  --r1-results "$PROJECT/experiments/klifs85_interaction/outputs/residual_kernel_fold1_v2/results.json" \
  --brics-checkpoint "42=$PROJECT/experiments/brics_hierarchical/outputs/graph_real_seed42/best.pt" \
  --brics-checkpoint "43=$PROJECT/experiments/brics_hierarchical/outputs/graph_real_seed43/best.pt" \
  --brics-checkpoint "44=$PROJECT/experiments/brics_hierarchical/outputs/graph_real_seed44/best.pt" \
  --noedge-checkpoint "$PROJECT/experiments/brics_hierarchical/outputs/graph_no_fragment_edges_seed42/best.pt" \
  --output-dir "$OUT/fold_1_development_reproduction" --device cuda:0 --batch-size 128 \
  --residual-mode in_sample > "$OUT/logs/fold1_reproduction.log" 2>&1

echo "STAGE=LABEL_INDEPENDENT_CACHES"
for FOLD in 2 3 4 5; do
  CKPT=$PROJECT/outputs/Refine_experiment/davis/cold_start/drug_cold_5fold_seed42/baseline/fold_${FOLD}/best_model.pt
  CACHE=$OUT/cache/fold_${FOLD}
  mkdir -p "$CACHE"
  "$PYTHON" "$SCRIPTS/n2_cache_label_free.py" \
    --project "$PROJECT" --checkpoint "$CKPT" \
    --output "$CACHE/global_with_features.pt" --device cuda:0 --batch-size 16 \
    > "$OUT/logs/cache_global_fold${FOLD}.log" 2>&1
  "$PYTHON" "$BASE_SCRIPTS/cache_brics_encoder_inputs.py" \
    --project "$PROJECT" --checkpoint "$CKPT" \
    --global-cache "$CACHE/global_with_features.pt" --mapping-cache "$MAPPING" \
    --drug-1d-dir "$PROJECT/data/processed/davis/drug_1d_chemberta2" \
    --drug-3d-dir "$PROJECT/data/processed/davis/drug_3d" --device cuda:0 \
    --output "$CACHE/drug_encoder_inputs.pt" \
    > "$OUT/logs/cache_drug_fold${FOLD}.log" 2>&1
  "$PYTHON" "$SCRIPTS/n2_augment_brics_chemistry.py" \
    --input-cache "$CACHE/drug_encoder_inputs.pt" --pairs-csv "$PAIRS" \
    --output "$CACHE/drug_encoder_inputs_chem.pt" \
    > "$OUT/logs/cache_chem_fold${FOLD}.log" 2>&1
done

common_train_args() {
  local fold=$1
  echo \
    --project "$PROJECT" \
    --fold "$fold" \
    --checkpoint "$PROJECT/outputs/Refine_experiment/davis/cold_start/drug_cold_5fold_seed42/baseline/fold_${fold}/best_model.pt" \
    --global-cache "$OUT/cache/fold_${fold}/global_with_features.pt" \
    --drug-cache "$OUT/cache/fold_${fold}/drug_encoder_inputs_chem.pt" \
    --outer-split "$PROJECT/data/splits/davis_drug_cold_5fold_seed42/fold_${fold}/split.json" \
    --inner-split "$OUT/splits/fold_${fold}/inner_split.json" \
    --pairs-csv "$PAIRS" --batch-size 128 --epochs 100 --patience 20 \
    --lr 3e-4 --weight-decay 1e-5 --lambda-align 0.05 \
    --lambda-reconstruct 0.02 --lambda-delta 0.001
}

echo "STAGE=SMOKE_TEST"
# Epoch 0 checks the complete label gate, cache/model path, split assertions,
# collate, forward pass, metric path and checkpoint serialization without
# consuming any outer-validation label.
SMOKE=$OUT/smoke/fold_2_real_seed42
# shellcheck disable=SC2046
"$PYTHON" "$SCRIPTS/n2_nested_train.py" $(common_train_args 2) \
  --phase inner --condition real --seed 42 --device cuda:0 --epochs 0 \
  --output-dir "$SMOKE" > "$OUT/logs/smoke.log" 2>&1
test -f "$SMOKE/best.pt"

run_fold() {
  local fold=$1
  local device=$2
  for seed in 42 43 44; do
    local base=$OUT/fold_${fold}/training/real_seed${seed}
    mkdir -p "$base/inner_selection_checkpoint" "$base/outer_refit_checkpoint"
    # shellcheck disable=SC2046
    "$PYTHON" "$SCRIPTS/n2_nested_train.py" $(common_train_args "$fold") \
      --phase inner --condition real --seed "$seed" --device "$device" \
      --output-dir "$base/inner_selection_checkpoint" \
      > "$OUT/logs/fold${fold}_real_seed${seed}_inner.log" 2>&1
    # shellcheck disable=SC2046
    "$PYTHON" "$SCRIPTS/n2_nested_train.py" $(common_train_args "$fold") \
      --phase refit --condition real --seed "$seed" --device "$device" \
      --selected-epoch-json "$base/inner_selection_checkpoint/result.json" \
      --output-dir "$base/outer_refit_checkpoint" \
      > "$OUT/logs/fold${fold}_real_seed${seed}_refit.log" 2>&1
  done
  local base=$OUT/fold_${fold}/training/noedge_seed42
  mkdir -p "$base/inner_selection_checkpoint" "$base/outer_refit_checkpoint"
  # shellcheck disable=SC2046
  "$PYTHON" "$SCRIPTS/n2_nested_train.py" $(common_train_args "$fold") \
    --phase inner --condition no_fragment_edges --seed 42 --device "$device" \
    --output-dir "$base/inner_selection_checkpoint" \
    > "$OUT/logs/fold${fold}_noedge_seed42_inner.log" 2>&1
  # shellcheck disable=SC2046
  "$PYTHON" "$SCRIPTS/n2_nested_train.py" $(common_train_args "$fold") \
    --phase refit --condition no_fragment_edges --seed 42 --device "$device" \
    --selected-epoch-json "$base/inner_selection_checkpoint/result.json" \
    --output-dir "$base/outer_refit_checkpoint" \
    > "$OUT/logs/fold${fold}_noedge_seed42_refit.log" 2>&1
}

echo "STAGE=LOCKED_16_RUNS"
run_fold 2 cuda:0 & P2=$!
run_fold 3 cuda:1 & P3=$!
run_fold 4 cuda:2 & P4=$!
run_fold 5 cuda:3 & P5=$!
wait "$P2" "$P3" "$P4" "$P5"

echo "STAGE=FINAL_ONLY_EVALUATION"
"$PYTHON" "$SCRIPTS/n2_evaluate_locked.py" \
  --project "$PROJECT" --output-root "$OUT" --pairs-csv "$PAIRS" \
  --similarity-pattern "$PROJECT/experiments/klifs85_interaction/data/similarity_audit_fold1/entity_similarities.npz" \
  --fold1-summary "$OUT/fold_1_development_reproduction/summary.json" \
  --device cuda:0 --batch-size 128 > "$OUT/logs/final_evaluation.log" 2>&1

echo "N2_COMPLETE=True"
