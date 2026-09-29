#!/usr/bin/env bash
set -euo pipefail

cd /data1/ztx/MyModel-MDTA

PYTHON=/data1/ztx/.conda/envs/mdta/bin/python
SCRIPT=experiments/brics_hierarchical/scripts/eval_brics_r1_complementarity.py
OUTPUT=experiments/brics_hierarchical/outputs/brics_r1_complementarity/fold_1
COMMON=(
  --project /data1/ztx/MyModel-MDTA
  --global-cache experiments/pdbbind_to_davis_transfer/data/global_predictions/fold1_with_features.pt
  --drug-cache experiments/brics_hierarchical/data/fold1_drug_encoder_inputs_chem.pt
  --similarity experiments/klifs85_interaction/data/similarity_audit_fold1/entity_similarities.npz
  --split data/splits/davis_drug_cold_5fold_seed42/fold_1/split.json
  --p13d-checkpoint outputs/Refine_experiment/davis/cold_start/drug_cold_5fold_seed42/baseline/fold_1/best_model.pt
  --r1-results experiments/klifs85_interaction/outputs/residual_kernel_fold1_v2/results.json
  --brics-checkpoint 42=experiments/brics_hierarchical/outputs/graph_real_seed42/best.pt
  --brics-checkpoint 43=experiments/brics_hierarchical/outputs/graph_real_seed43/best.pt
  --brics-checkpoint 44=experiments/brics_hierarchical/outputs/graph_real_seed44/best.pt
  --noedge-checkpoint experiments/brics_hierarchical/outputs/graph_no_fragment_edges_seed42/best.pt
  --output-dir "$OUTPUT"
  --device cuda:0
  --batch-size 128
  --residual-mode in_sample
)

"$PYTHON" -m py_compile "$SCRIPT"
"$PYTHON" "$SCRIPT" "${COMMON[@]}" --sanity-only
"$PYTHON" "$SCRIPT" "${COMMON[@]}"
