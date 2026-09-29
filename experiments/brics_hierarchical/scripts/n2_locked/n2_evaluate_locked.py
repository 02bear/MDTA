#!/usr/bin/env python3
"""Final-only evaluation for the preregistered N2 locked multi-fold validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def selected_labels(pairs_csv: Path, allowed_rows, total_rows):
    allowed = set(map(int, allowed_rows))
    output = np.full(total_rows, np.nan, dtype=np.float64)
    seen = set()
    with pairs_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index in allowed:
                output[index] = float(row["label"])
                seen.add(index)
    if seen != allowed:
        raise RuntimeError("not all permitted labels were found")
    return output


def build_grid(global_data, similarity_data, split):
    drug_ids = [str(x) for x in similarity_data["drug_ids"].tolist()]
    protein_ids = [str(x) for x in similarity_data["protein_ids"].tolist()]
    drug_lookup = {x: i for i, x in enumerate(drug_ids)}
    protein_lookup = {x: i for i, x in enumerate(protein_ids)}
    row_index = np.full((len(drug_ids), len(protein_ids)), -1, dtype=np.int64)
    for row, (drug, protein) in enumerate(zip(global_data["drug_id"], global_data["protein_id"])):
        row_index[drug_lookup[str(drug)], protein_lookup[str(protein)]] = row
    if np.any(row_index < 0):
        raise RuntimeError("incomplete Davis grid")
    train = np.asarray([drug_lookup[str(x)] for x in split["train_drugs"]], dtype=int)
    val = np.asarray([drug_lookup[str(x)] for x in split["val_drugs"]], dtype=int)
    if set(train) & set(val):
        raise RuntimeError("outer train/validation drug overlap")
    if not np.array_equal(np.sort(row_index[train].ravel()), np.sort(split["train_indices"])):
        raise RuntimeError("outer train indices mismatch")
    if not np.array_equal(np.sort(row_index[val].ravel()), np.sort(split["val_indices"])):
        raise RuntimeError("outer validation indices mismatch")
    return drug_ids, protein_ids, row_index, train, val


def infer(graph, global_data, drug_data, rows, condition, seed, checkpoint,
          project, p13d_checkpoint, device, batch_size):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    store = graph.ChemStore(global_data, drug_data, condition, seed)
    loader = DataLoader(graph.base.IndexDataset(rows), batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=store.collate)
    model = graph.BRICSChemGraphP13D(project, p13d_checkpoint).to(device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = saved.get("result", {})
    required_false = result.get("INNER_WEIGHTS_REUSED") is False
    if not required_false:
        raise RuntimeError(f"fresh-start guard missing or false: {checkpoint}")
    incompat = model.load_state_dict(saved["hierarchy_state"], strict=False)
    invalid_missing = [k for k in incompat.missing_keys if not (k.startswith("drug_fusion.") or k.startswith("decoder."))]
    if incompat.unexpected_keys or invalid_missing:
        raise RuntimeError(f"checkpoint incompatibility: {checkpoint}")
    observed = graph.base.evaluate(model, loader, device)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return observed["index"].numpy(), observed["prediction"].numpy().astype(np.float64), result


def apply_r1(r1, query, train, labels, base, similarity, config):
    residual = labels - base
    prediction, delta, alpha = r1.outer_predictions(
        query, train, residual, base, similarity, config, r2_best=None)
    return prediction, delta, alpha


def bootstrap_drugs(deltas, n=20000, seed=20260902):
    deltas = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(deltas), size=(n, len(deltas)))
    samples = deltas[draws].mean(axis=1)
    return {
        "definition": "BRICS+R1 MSE - R1 MSE; negative favors BRICS+R1",
        "cluster_unit": "confirmation drug with all 442 protein pairs; seed MSE averaged within drug",
        "replicates": n, "seed": seed, "clusters": len(deltas),
        "mean_delta_mse": float(deltas.mean()),
        "bootstrap_95_ci": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
        "probability_candidate_better": float(np.mean(samples < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--similarity-pattern", required=True)
    parser.add_argument("--fold1-summary", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    scripts = args.project / "experiments/brics_hierarchical/scripts"
    r1_dir = args.project / "experiments/klifs85_interaction"
    sys.path.insert(0, str(scripts)); sys.path.insert(0, str(r1_dir)); sys.path.insert(0, str(args.project))
    import train_brics_graph_stage1 as graph
    from experiments.klifs85_interaction import run_residual_kernel as r1
    from experiments.klifs85_interaction.train_klifs_interact import metrics

    # Gate 1: all 16 formal final checkpoints must exist and carry the fresh-start
    # declaration before any confirmation label field is parsed.
    checkpoint_manifest = {}
    for fold in range(2, 6):
        entries = {}
        for seed in SEEDS:
            path = args.output_root / f"fold_{fold}/training/real_seed{seed}/outer_refit_checkpoint/final.pt"
            if not path.exists(): raise FileNotFoundError(path)
            entries[f"real_seed{seed}"] = {"path": str(path), "sha256": sha256(path)}
        path = args.output_root / f"fold_{fold}/training/noedge_seed42/outer_refit_checkpoint/final.pt"
        if not path.exists(): raise FileNotFoundError(path)
        entries["noedge_seed42"] = {"path": str(path), "sha256": sha256(path)}
        checkpoint_manifest[str(fold)] = entries
    manifest_path = args.output_root / "evaluation/frozen_checkpoint_manifest.json"
    write_json(manifest_path, {
        "all_16_final_checkpoints_present": True,
        "confirmation_labels_accessed_before_manifest": False,
        "checkpoints": checkpoint_manifest,
    })
    print("ALL_16_FINAL_CHECKPOINTS_FROZEN = True", flush=True)
    print("CONFIRMATION_LABELS_ACCESSED_BEFORE_FINAL = False", flush=True)

    fold_results, all_drug_delta = {}, []
    pooled_r1_sse = pooled_combo_sse = pooled_count = 0
    pooled_noedge_sse = 0
    for fold in range(2, 6):
        fold_dir = args.output_root / f"fold_{fold}"
        global_path = args.output_root / f"cache/fold_{fold}/global_with_features.pt"
        drug_path = args.output_root / f"cache/fold_{fold}/drug_encoder_inputs_chem.pt"
        split_path = args.project / f"data/splits/davis_drug_cold_5fold_seed42/fold_{fold}/split.json"
        p13d_path = args.project / f"outputs/Refine_experiment/davis/cold_start/drug_cold_5fold_seed42/baseline/fold_{fold}/best_model.pt"
        r1_path = args.project / f"experiments/klifs85_interaction/outputs/residual_kernel_fold{fold}_v2/results.json"
        similarity_path = Path(args.similarity_pattern.format(fold=fold))
        global_data = torch.load(global_path, map_location="cpu", weights_only=False)
        if "label" in global_data: raise RuntimeError("label leaked into N2 cache")
        drug_data = torch.load(drug_path, map_location="cpu", weights_only=False)
        similarity = np.load(similarity_path, allow_pickle=True)
        split = json.loads(split_path.read_text(encoding="utf-8"))
        drug_ids, protein_ids, row_index, train, val = build_grid(global_data, similarity, split)

        # Gate 2: after the 16-checkpoint manifest is frozen, parse labels only for
        # this fold's outer-train and outer-validation rows; test rows stay NaN.
        permitted = np.concatenate([row_index[train].ravel(), row_index[val].ravel()])
        row_labels = selected_labels(args.pairs_csv, permitted, len(global_data["prediction"]))
        gated = dict(global_data); gated["label"] = torch.from_numpy(row_labels).float()
        label_grid = row_labels[row_index]
        p13d_grid = global_data["prediction"].numpy().astype(np.float64)[row_index]
        label_val, p13d_val = label_grid[val], p13d_grid[val]
        r1_saved = json.loads(r1_path.read_text(encoding="utf-8"))
        selected = r1_saved["inner"]["R1_best"]
        config = {k: selected[k] for k in ("gamma", "k_drug", "min_drug_similarity", "tau", "beta", "clip", "scale")}
        drug_similarity = similarity["drug_similarity"].astype(np.float64)
        r1_val, _, _ = apply_r1(r1, val, train, label_grid, p13d_grid, drug_similarity, config)
        r1_metrics = metrics(label_val.ravel(), r1_val.ravel())

        rows = np.concatenate([row_index[train].ravel(), row_index[val].ravel()]).astype(int)
        noedge_path = Path(checkpoint_manifest[str(fold)]["noedge_seed42"]["path"])
        idx, pred, noedge_run = infer(graph, gated, drug_data, rows, "no_fragment_edges", 42,
                                      noedge_path, args.project, p13d_path, args.device, args.batch_size)
        noedge_flat = np.full(len(row_labels), np.nan); noedge_flat[idx] = pred
        noedge_grid = noedge_flat[row_index]
        noedge_r1, _, _ = apply_r1(r1, val, train, label_grid, noedge_grid, drug_similarity, config)
        noedge_metrics = metrics(label_val.ravel(), noedge_r1.ravel())

        seed_results, combo_predictions = {}, []
        per_drug_seed_mse = []
        for seed in SEEDS:
            path = Path(checkpoint_manifest[str(fold)][f"real_seed{seed}"]["path"])
            idx, pred, run = infer(graph, gated, drug_data, rows, "real", seed, path,
                                   args.project, p13d_path, args.device, args.batch_size)
            flat = np.full(len(row_labels), np.nan); flat[idx] = pred
            brics_grid = flat[row_index]
            combo, _, _ = apply_r1(r1, val, train, label_grid, brics_grid, drug_similarity, config)
            brics_m = metrics(label_val.ravel(), brics_grid[val].ravel())
            combo_m = metrics(label_val.ravel(), combo.ravel())
            epoch0_validation = None
            if int(run["selected_epoch"]) == 0:
                max_abs = float(np.max(np.abs(brics_grid[val] - p13d_val)))
                epoch0_validation = {
                    "max_abs_pred_epoch0_vs_p13d": max_abs,
                    "mse_epoch0": brics_m["mse"],
                    "prediction_equivalent": bool(max_abs <= 1e-5),
                    "name": "P13D baseline" if max_abs <= 1e-5 else "no-training initialized BRICS model",
                }
            seed_results[str(seed)] = {
                "E_star": int(run["selected_epoch"]), "brics": brics_m, "brics_r1": combo_m,
                "epoch0_validation_audit": epoch0_validation,
                "checkpoint": str(path), "checkpoint_sha256": sha256(path),
            }
            combo_predictions.append(combo)
            per_drug_seed_mse.append(np.mean((combo - label_val) ** 2, axis=1))

        real_fold_mse = float(np.mean([seed_results[str(s)]["brics_r1"]["mse"] for s in SEEDS]))
        per_drug_combo = np.mean(np.stack(per_drug_seed_mse), axis=0)
        per_drug_r1 = np.mean((r1_val - label_val) ** 2, axis=1)
        deltas = per_drug_combo - per_drug_r1
        all_drug_delta.extend(deltas.tolist())
        val_drug_ids = [drug_ids[x] for x in val]
        per_drug = pd.DataFrame({
            "fold": fold, "drug_id": val_drug_ids, "pairs": len(protein_ids),
            "r1_mse": per_drug_r1, "real_brics_r1_seedmean_mse": per_drug_combo,
            "delta_combo_minus_r1": deltas, "improved": deltas < 0,
            "noedge_r1_mse": np.mean((noedge_r1 - label_val) ** 2, axis=1),
        })
        (fold_dir / "evaluation").mkdir(parents=True, exist_ok=True)
        per_drug.to_csv(fold_dir / "evaluation/per_drug.csv", index=False)
        fold_result = {
            "scope": "locked outer-fold confirmation",
            "statistical_qualification": "not fully independent entity-level confirmation",
            "fold": fold, "r1": r1_metrics, "seeds": seed_results,
            "noedge_r1": noedge_metrics, "real_brics_r1_seedmean_mse": real_fold_mse,
            "real_brics_r1_better_than_r1": bool(real_fold_mse < r1_metrics["mse"]),
            "real_brics_r1_better_than_noedge_r1": bool(real_fold_mse < noedge_metrics["mse"]),
            "improved_drugs": int(np.sum(deltas < 0)), "validation_drugs": len(deltas),
            "r1_config": config,
        }
        write_json(fold_dir / "evaluation/metrics.json", fold_result)
        fold_results[str(fold)] = fold_result
        for combo in combo_predictions:
            pooled_combo_sse += float(np.square(combo - label_val).sum())
            pooled_r1_sse += float(np.square(r1_val - label_val).sum())
            pooled_noedge_sse += float(np.square(noedge_r1 - label_val).sum())
            pooled_count += int(label_val.size)

    fold_combo = [fold_results[str(f)]["real_brics_r1_seedmean_mse"] for f in range(2, 6)]
    fold_r1 = [fold_results[str(f)]["r1"]["mse"] for f in range(2, 6)]
    fold_noedge = [fold_results[str(f)]["noedge_r1"]["mse"] for f in range(2, 6)]
    improved_folds = sum(x < y for x, y in zip(fold_combo, fold_r1))
    topology_folds = sum(x < y for x, y in zip(fold_combo, fold_noedge))
    macro_combo, macro_r1 = float(np.mean(fold_combo)), float(np.mean(fold_r1))
    pooled_combo, pooled_r1 = pooled_combo_sse / pooled_count, pooled_r1_sse / pooled_count
    majority_drugs = int(np.sum(np.asarray(all_drug_delta) < 0)) > len(all_drug_delta) / 2
    criteria = {
        "at_least_3_of_4_confirmation_folds_combo_better_r1": improved_folds >= 3,
        "fold2_5_macro_mean_combo_better_r1": macro_combo < macro_r1,
        "fold2_5_pooled_combo_better_r1": pooled_combo < pooled_r1,
        "majority_confirmation_drugs_improved": majority_drugs,
        "topology_desideratum_realedge_better_noedge_at_least_3_of_4": topology_folds >= 3,
    }
    efficacy = all(list(criteria.values())[:4])
    topology = criteria["topology_desideratum_realedge_better_noedge_at_least_3_of_4"]
    verdict = "SUPPORTED_WITH_TOPOLOGY" if efficacy and topology else (
        "SUPPORTED_WITHOUT_TOPOLOGY_SPECIFIC_EVIDENCE" if efficacy else "NOT_SUPPORTED")
    bootstrap = bootstrap_drugs(all_drug_delta)
    aggregate = {
        "scope": "locked multi-fold validation",
        "qualification": (
            "Fold2-Fold5 are locked outer-fold confirmations, not fully independent entity-level "
            "confirmation: some validation drugs occurred in training sets of other folds, including Fold1 development. "
            "This is not within-fold leakage."
        ),
        "folds": fold_results,
        "macro": {"brics_r1": macro_combo, "r1": macro_r1, "noedge_r1": float(np.mean(fold_noedge))},
        "pooled": {"brics_r1": pooled_combo, "r1": pooled_r1, "noedge_r1": pooled_noedge_sse / pooled_count},
        "improved_folds": improved_folds, "topology_folds": topology_folds,
        "improved_confirmation_drugs": int(np.sum(np.asarray(all_drug_delta) < 0)),
        "total_confirmation_drugs": len(all_drug_delta), "cluster_bootstrap": bootstrap,
        "primary_criteria": criteria, "primary_confirmation_verdict": verdict,
        "Fold1_in_primary_confirmation": False,
        "confirmation_outer_validation_labels_never_accessed_before_final": True,
        "checkpoint_manifest": str(manifest_path),
    }
    if args.fold1_summary and args.fold1_summary.exists():
        fold1 = json.loads(args.fold1_summary.read_text(encoding="utf-8"))
        aggregate["descriptive_fold1"] = {
            "scope": "development/descriptive only", "source": str(args.fold1_summary),
            "decision": fold1.get("decision"), "reproduction": fold1.get("reproduction"),
        }
    write_json(args.output_root / "evaluation/final_summary.json", aggregate)

    lines = [
        "# N2 Locked Multi-fold Validation Report", "",
        "## Statistical scope", "",
        aggregate["qualification"], "",
        "Fold1 is development/descriptive only and is excluded from every primary criterion.", "",
        "## Fold2–Fold5 results", "",
        "| Fold | R1 MSE | RealEdge+R1 seed-mean MSE | NoEdge+R1 MSE | Real<R1 | Real<NoEdge | Improved drugs |",
        "|---:|---:|---:|---:|:---:|:---:|---:|",
    ]
    for fold in range(2, 6):
        x = fold_results[str(fold)]
        lines.append(f"| {fold} | {x['r1']['mse']:.6f} | {x['real_brics_r1_seedmean_mse']:.6f} | {x['noedge_r1']['mse']:.6f} | {x['real_brics_r1_better_than_r1']} | {x['real_brics_r1_better_than_noedge_r1']} | {x['improved_drugs']}/{x['validation_drugs']} |")
    lines += ["", "## Primary confirmation verdict", "", f"**{verdict}**", "", "```json", json.dumps(criteria, indent=2), "```", "",
              f"Macro MSE: BRICS+R1={macro_combo:.6f}, R1={macro_r1:.6f}.",
              f"Pooled MSE: BRICS+R1={pooled_combo:.6f}, R1={pooled_r1:.6f}.",
              f"Drug-cluster bootstrap 95% CI: {bootstrap['bootstrap_95_ci']}.", "",
              "Detailed fold×seed MSE/CI/Rm2, E*, epoch-0 audits, and per-drug CSV files are stored beside this report.", ""]
    report = args.output_root / "evaluation/N2_FINAL_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(report), "verdict": verdict, "criteria": criteria}, indent=2), flush=True)


if __name__ == "__main__":
    main()
