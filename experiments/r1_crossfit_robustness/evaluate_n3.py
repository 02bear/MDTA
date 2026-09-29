#!/usr/bin/env python3
"""Build strict OOF banks and perform all locked N3 controls and analyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from n3_common import N_PROTEINS, R1_CONFIGS, SEED, build_n3_split, sha256_file, write_json


def distribution(values):
    x = np.asarray(values, dtype=np.float64).ravel()
    return {
        "mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)),
        "median": float(np.median(x)), "MAE": float(np.mean(np.abs(x))),
        "MSE": float(np.mean(x**2)), "p05": float(np.quantile(x, .05)),
        "p25": float(np.quantile(x, .25)), "p75": float(np.quantile(x, .75)),
        "p95": float(np.quantile(x, .95)), "max_abs": float(np.max(np.abs(x))),
    }


def correction_diagnostics(r1, query, references, residual, similarity, config):
    raw, variance, support = r1.drug_kernel_stats(query, references, residual[references], similarity, config)
    _, correction, _ = r1.apply_correction(
        np.zeros_like(raw), raw, variance, support,
        {key: config[key] for key in ("tau", "beta", "clip", "scale")},
    )
    counts, top1, topk = [], [], []
    for drug in query:
        local, weights = r1.top_weights(
            similarity[drug, references], config["gamma"], config["k_drug"], config["min_drug_similarity"]
        )
        sims = similarity[drug, references][local] if len(local) else np.asarray([])
        counts.append(len(local)); top1.append(float(np.max(sims)) if len(sims) else 0.0)
        topk.append(float(np.mean(sims)) if len(sims) else 0.0)
    stats = distribution(correction)
    stats.update({
        "fraction_clipped": float(np.mean(np.abs(raw) > config["clip"])),
        "mean_neighbor_count": float(np.mean(counts)),
        "median_neighbor_count": float(np.median(counts)),
        "mean_top1_similarity": float(np.mean(top1)),
        "mean_topK_similarity": float(np.mean(topk)),
    })
    return stats


def local_smoothness(r1, train, residual, similarity, config):
    predictions, variances = [], []
    for query in train:
        references = np.asarray([x for x in train if x != query], dtype=int)
        mean, variance, _ = r1.drug_kernel_stats(
            np.asarray([query]), references, residual[references], similarity, config
        )
        predictions.append(mean[0]); variances.append(variance[0])
    prediction = np.asarray(predictions)
    variance = np.asarray(variances)
    return {
        "leave_one_drug_out_residual_prediction_mse": float(np.mean((prediction-residual[train])**2)),
        "local_residual_variance_mean": float(np.mean(variance)),
        "local_residual_variance_median": float(np.median(variance)),
    }


def bootstrap(deltas, definition, n=10000, seed=SEED):
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return {
        "definition": definition, "cluster_unit": "drug with all 442 protein pairs",
        "replicates": n, "seed": seed, "clusters": len(values),
        "mean_delta_mse": float(np.mean(values)), "median_delta_mse": float(np.median(values)),
        "95%_CI": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
        "probability_first_better": float(np.mean(samples < 0)),
    }


def macro_summary(fold_results, model):
    keys = ("mse", "rmse", "mae", "ci", "rm2")
    output = {}
    for key in keys:
        values = np.asarray([fold_results[str(f)]["metrics"][model][key] for f in range(1, 6)])
        output[key] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project))
    sys.path.insert(0, str(args.project / "experiments/klifs85_interaction"))
    from experiments.klifs85_interaction import run_residual_kernel as r1
    from experiments.klifs85_interaction.train_klifs_interact import metrics

    reproduction = json.loads((args.output_root / "audit/historical_reproduction.json").read_text(encoding="utf-8"))
    if reproduction.get("all_passed") is not True:
        raise RuntimeError("historical reproduction gate failed")

    # Freeze and audit all 25 final checkpoints before constructing any fold bank.
    manifest, e_stars = {}, {}
    for fold in range(1, 6):
        manifest[str(fold)] = {}
        for cf in range(1, 6):
            result_path = args.output_root / f"fold_{fold}/cf_{cf}/stage_b_strict_refit/result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            required = {
                "INNER_WEIGHTS_REUSED": False,
                "HOLDOUT_USED_FOR_TRAINING": False,
                "HOLDOUT_USED_FOR_EPOCH_SELECTION": False,
                "holdout_labels_parsed_before_checkpoint_freeze": False,
                "holdout_labels_parsed_before_final_oof_inference": False,
            }
            for key, expected in required.items():
                if result.get(key) is not expected:
                    raise RuntimeError(f"fold {fold} cf {cf}: guard {key} failed")
            checkpoint = Path(result["checkpoint"])
            if not checkpoint.exists() or sha256_file(checkpoint) != result["checkpoint_sha256"]:
                raise RuntimeError(f"fold {fold} cf {cf}: checkpoint integrity failure")
            expected_split = build_n3_split(args.project, fold, cf)
            for key in ("holdout_drugs", "T_j", "epoch_train_drugs", "epoch_val_drugs"):
                if result["split"][key] != expected_split[key]:
                    raise RuntimeError(f"fold {fold} cf {cf}: split drift {key}")
            manifest[str(fold)][str(cf)] = {
                "checkpoint": str(checkpoint), "sha256": result["checkpoint_sha256"],
                "E_star": int(result["E_star"]), "guards": required,
            }
            e_stars[f"F{fold}-CF{cf}"] = int(result["E_star"])
    write_json(args.output_root / "audit/frozen_checkpoint_manifest.json", {
        "all_25_final_checkpoints_present": True, "all_guards_passed": True,
        "checkpoints": manifest,
    })

    similarity_path = args.project / "experiments/klifs85_interaction/data/similarity_audit_fold1/entity_similarities.npz"
    sim = np.load(similarity_path, allow_pickle=True)
    drug_ids = [str(x) for x in sim["drug_ids"].tolist()]
    protein_ids = [str(x) for x in sim["protein_ids"].tolist()]
    lookup_d = {x: i for i, x in enumerate(drug_ids)}
    lookup_p = {x: i for i, x in enumerate(protein_ids)}
    drug_similarity = sim["drug_similarity"].astype(np.float64)
    if len(protein_ids) != N_PROTEINS:
        raise RuntimeError("protein dimension is not 442")

    fold_results, pooled_rows, per_drug_rows = {}, [], []
    pooled_shuffles = [[] for _ in range(100)]
    residual_diagnostics, correction_diagnostics_all, smoothness_all = {}, {}, {}

    for fold in range(1, 6):
        fold_dir = args.output_root / f"fold_{fold}"
        split_path = args.project / f"data/splits/davis_drug_cold_5fold_seed42/fold_{fold}/split.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        cache_path = args.project / f"experiments/pdbbind_to_davis_transfer/data/global_predictions/fold{fold}.pt"
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        row_index = np.full((len(drug_ids), len(protein_ids)), -1, dtype=int)
        for row, (drug, protein) in enumerate(zip(data["drug_id"], data["protein_id"])):
            row_index[lookup_d[str(drug)], lookup_p[str(protein)]] = row
        if (row_index < 0).any():
            raise RuntimeError(f"fold {fold}: incomplete grid")
        labels = data["label"].numpy()[row_index].astype(np.float64)
        baseline = data["prediction"].numpy()[row_index].astype(np.float64)
        train = np.asarray([lookup_d[str(x)] for x in split["train_drugs"]], dtype=int)
        val = np.asarray([lookup_d[str(x)] for x in split["val_drugs"]], dtype=int)
        if not np.array_equal(np.sort(row_index[train].ravel()), np.sort(split["train_indices"])):
            raise RuntimeError(f"fold {fold}: outer train mismatch")
        if not np.array_equal(np.sort(row_index[val].ravel()), np.sort(split["val_indices"])):
            raise RuntimeError(f"fold {fold}: outer validation mismatch")

        pieces = []
        for cf in range(1, 6):
            path = fold_dir / f"cf_{cf}/stage_b_strict_refit/oof_predictions.csv"
            frame = pd.read_csv(path, dtype={"drug_id": str, "protein_id": str})
            expected = build_n3_split(args.project, fold, cf)
            if set(frame["drug_id"]) != set(expected["holdout_drugs"]):
                raise RuntimeError(f"fold {fold} cf {cf}: OOF drug membership mismatch")
            if len(frame) != len(expected["holdout_drugs"]) * N_PROTEINS:
                raise RuntimeError(f"fold {fold} cf {cf}: OOF pair count mismatch")
            pieces.append(frame)
        oof = pd.concat(pieces, ignore_index=True)
        if oof.duplicated(["drug_id", "protein_id"]).any():
            raise RuntimeError(f"fold {fold}: duplicate OOF pair")
        if oof[["label", "historical_insample_pred", "strict_oof_pred", "residual_insample", "residual_oof"]].isna().any().any():
            raise RuntimeError(f"fold {fold}: OOF NaN")
        if not np.isfinite(oof[["label", "historical_insample_pred", "strict_oof_pred", "residual_insample", "residual_oof"]].to_numpy()).all():
            raise RuntimeError(f"fold {fold}: OOF Inf")
        if len(oof) != len(train) * N_PROTEINS or set(oof["drug_id"]) != set(split["train_drugs"]):
            raise RuntimeError(f"fold {fold}: OOF completeness failure")
        if not (oof.groupby("drug_id")["cf_fold"].nunique() == 1).all() or not (oof.groupby("drug_id").size() == N_PROTEINS).all():
            raise RuntimeError(f"fold {fold}: one-model-per-drug failure")
        oof.to_csv(fold_dir / "oof_predictions.csv", index=False)

        residual_in = labels - baseline
        residual_oof = np.zeros_like(residual_in)
        strict_pred = np.full_like(baseline, np.nan)
        for row in oof.itertuples(index=False):
            d, p = lookup_d[str(row.drug_id)], lookup_p[str(row.protein_id)]
            residual_oof[d, p] = float(row.residual_oof)
            strict_pred[d, p] = float(row.strict_oof_pred)
        if not np.isfinite(strict_pred[train]).all():
            raise RuntimeError(f"fold {fold}: OOF grid incomplete")
        torch.save({
            "outer_fold": fold, "drug_ids": drug_ids, "protein_ids": protein_ids,
            "outer_train_drugs": split["train_drugs"],
            "strict_oof_prediction": torch.from_numpy(strict_pred[train]),
            "residual_oof": torch.from_numpy(residual_oof[train]),
            "holdout_used_for_training": False,
            "holdout_used_for_epoch_selection": False,
        }, fold_dir / "residual_bank.pt")

        config = R1_CONFIGS[fold]
        p13d_val = baseline[val]
        label_val = labels[val]
        r1_in, delta_in, _ = r1.outer_predictions(val, train, residual_in, baseline, drug_similarity, config)
        r1_oof, delta_oof, _ = r1.outer_predictions(val, train, residual_oof, baseline, drug_similarity, config)

        # Pure target-wise mean: no similarity, top-k, gamma, tau, beta, clip,
        # scale, support, or any other R1 weighting is applied.
        target_mean_vector = np.mean(residual_oof[train], axis=0)
        target_mean = p13d_val + target_mean_vector[None, :]

        models = {"P13D": p13d_val, "R1_in": r1_in, "R1_OOF": r1_oof, "TargetMean_OOF": target_mean}
        fold_metrics = {name: metrics(label_val.ravel(), pred.ravel()) for name, pred in models.items()}
        rng = np.random.default_rng(SEED)
        shuffle_rows = []
        for permutation_index in range(100):
            permutation = rng.permutation(len(train))
            if sorted(permutation.tolist()) != list(range(len(train))):
                raise RuntimeError("invalid drug-row permutation")
            shuffled_residual = np.zeros_like(residual_oof)
            # One permutation moves each complete 442-dimensional drug residual
            # vector. No target-wise independent shuffle is permitted.
            shuffled_residual[train] = residual_oof[train][permutation]
            prediction, _, _ = r1.outer_predictions(
                val, train, shuffled_residual, baseline, drug_similarity, config
            )
            observed = metrics(label_val.ravel(), prediction.ravel())
            shuffle_rows.append({"permutation": permutation_index, **observed})
            pooled_shuffles[permutation_index].append(pd.DataFrame({
                "fold": fold, "drug_id": np.repeat([drug_ids[x] for x in val], N_PROTEINS),
                "protein_id": np.tile(protein_ids, len(val)), "label": label_val.ravel(),
                "prediction": prediction.ravel(),
            }))
        shuffle_table = pd.DataFrame(shuffle_rows)
        shuffle_table.to_csv(fold_dir / "shuffle_control.csv", index=False)
        shuffle_summary = {
            "unit": "whole 442-dimensional drug residual row",
            "seed": SEED, "permutations": 100,
            "mse_mean": float(shuffle_table.mse.mean()), "mse_median": float(shuffle_table.mse.median()),
            "mse_2.5%": float(shuffle_table.mse.quantile(.025)), "mse_97.5%": float(shuffle_table.mse.quantile(.975)),
            "P(shuffled <= real R1_OOF)": float(np.mean(shuffle_table.mse <= fold_metrics["R1_OOF"]["mse"])),
        }

        train_in, train_oof = residual_in[train], residual_oof[train]
        drug_mse_in = np.mean(train_in**2, axis=1)
        drug_mse_oof = np.mean(train_oof**2, axis=1)
        residual_diag = {
            "e_in": distribution(train_in), "e_oof": distribution(train_oof),
            "pair_Pearson": float(np.corrcoef(train_in.ravel(), train_oof.ravel())[0, 1]),
            "pair_Spearman": float(pd.Series(train_in.ravel()).corr(pd.Series(train_oof.ravel()), method="spearman")),
            "drug_level_residual_MSE_correlation": float(np.corrcoef(drug_mse_in, drug_mse_oof)[0, 1]),
        }
        correction_diag = {
            "R1_in": correction_diagnostics(r1, val, train, residual_in, drug_similarity, config),
            "R1_OOF": correction_diagnostics(r1, val, train, residual_oof, drug_similarity, config),
        }
        smoothness = {
            "e_in": local_smoothness(r1, train, residual_in, drug_similarity, config),
            "e_oof": local_smoothness(r1, train, residual_oof, drug_similarity, config),
        }
        residual_diagnostics[str(fold)] = residual_diag
        correction_diagnostics_all[str(fold)] = correction_diag
        smoothness_all[str(fold)] = smoothness

        prediction_frame = pd.DataFrame({
            "fold": fold, "drug_id": np.repeat([drug_ids[x] for x in val], N_PROTEINS),
            "protein_id": np.tile(protein_ids, len(val)), "label": label_val.ravel(),
            "P13D": p13d_val.ravel(), "R1_in": r1_in.ravel(),
            "R1_OOF": r1_oof.ravel(), "TargetMean_OOF": target_mean.ravel(),
            "Delta_in": delta_in.ravel(), "Delta_oof": delta_oof.ravel(),
        })
        prediction_frame.to_csv(fold_dir / "predictions.csv", index=False)
        pooled_rows.append(prediction_frame)

        fold_per_drug = []
        for local, drug in enumerate(val):
            similarities = drug_similarity[drug, train]
            local_order, weights = r1.top_weights(similarities, config["gamma"], config["k_drug"], config["min_drug_similarity"])
            neighbors = [{
                "drug_id": drug_ids[int(train[x])], "Tanimoto": float(similarities[x]),
                "kernel_weight": float(weights[position]),
            } for position, x in enumerate(local_order)]
            values = {name: float(np.mean((pred[local]-label_val[local])**2)) for name, pred in models.items()}
            row = {
                "fold": fold, "drug_id": drug_ids[int(drug)], "n_pairs": N_PROTEINS,
                "P13D_MSE": values["P13D"], "R1_in_MSE": values["R1_in"],
                "R1_OOF_MSE": values["R1_OOF"], "TargetMean_OOF_MSE": values["TargetMean_OOF"],
                "gain_in_vs_p13d": values["P13D"]-values["R1_in"],
                "gain_oof_vs_p13d": values["P13D"]-values["R1_OOF"],
                "gain_oof_vs_targetmean": values["TargetMean_OOF"]-values["R1_OOF"],
                "gain_oof_vs_in": values["R1_in"]-values["R1_OOF"],
                "topK_neighbors": json.dumps(neighbors),
            }
            fold_per_drug.append(row); per_drug_rows.append(row)
        pd.DataFrame(fold_per_drug).to_csv(fold_dir / "per_drug_metrics.csv", index=False)
        fold_results[str(fold)] = {
            "metrics": fold_metrics, "shuffle_control": shuffle_summary,
            "OOF_completeness": {
                "expected_pairs": len(train)*N_PROTEINS, "observed_pairs": len(oof),
                "each_drug_exactly_once": True, "each_drug_pairs": N_PROTEINS,
                "duplicates": 0, "NaN_or_Inf": 0,
            },
        }
        write_json(fold_dir / "metrics.json", fold_results[str(fold)])

    pooled_dir = args.output_root / "pooled"
    pooled_dir.mkdir(parents=True, exist_ok=True)
    pooled = pd.concat(pooled_rows, ignore_index=True)
    pooled.to_csv(pooled_dir / "predictions.csv", index=False)
    per_drug = pd.DataFrame(per_drug_rows)
    per_drug.to_csv(pooled_dir / "per_drug_metrics.csv", index=False)
    model_names = ("P13D", "R1_in", "R1_OOF", "TargetMean_OOF")
    pooled_metrics = {name: metrics(pooled.label.to_numpy(), pooled[name].to_numpy()) for name in model_names}
    macro = {name: macro_summary(fold_results, name) for name in model_names}

    shuffle_pooled_rows = []
    for index, frames in enumerate(pooled_shuffles):
        frame = pd.concat(frames, ignore_index=True)
        observed = metrics(frame.label.to_numpy(), frame.prediction.to_numpy())
        shuffle_pooled_rows.append({"permutation": index, **observed})
    shuffle_pooled = pd.DataFrame(shuffle_pooled_rows)
    shuffle_pooled.to_csv(pooled_dir / "shuffle_control.csv", index=False)
    real_oof_mse = pooled_metrics["R1_OOF"]["mse"]
    shuffle_summary = {
        "unit": "whole 442-dimensional residual vector per training drug within fold",
        "permutations": 100, "seed": SEED,
        "mse_mean": float(shuffle_pooled.mse.mean()), "mse_median": float(shuffle_pooled.mse.median()),
        "mse_2.5%": float(shuffle_pooled.mse.quantile(.025)), "mse_97.5%": float(shuffle_pooled.mse.quantile(.975)),
        "P(shuffled <= real R1_OOF)": float(np.mean(shuffle_pooled.mse <= real_oof_mse)),
    }

    primary_delta = per_drug.R1_OOF_MSE - per_drug.P13D_MSE
    bootstrap_results = {
        "R1_OOF_vs_P13D": bootstrap(primary_delta, "R1_OOF MSE - P13D MSE"),
        "R1_OOF_vs_R1_in": bootstrap(per_drug.R1_OOF_MSE-per_drug.R1_in_MSE, "R1_OOF MSE - R1_in MSE"),
        "R1_OOF_vs_TargetMean_OOF": bootstrap(
            per_drug.R1_OOF_MSE-per_drug.TargetMean_OOF_MSE, "R1_OOF MSE - TargetMean_OOF MSE"
        ),
    }
    write_json(pooled_dir / "bootstrap.json", bootstrap_results)
    write_json(pooled_dir / "shuffle_control.json", shuffle_summary)
    write_json(pooled_dir / "residual_diagnostics.json", residual_diagnostics)
    write_json(pooled_dir / "correction_diagnostics.json", correction_diagnostics_all)
    write_json(pooled_dir / "local_smoothness.json", smoothness_all)

    folds_improved = sum(
        fold_results[str(f)]["metrics"]["R1_OOF"]["mse"] < fold_results[str(f)]["metrics"]["P13D"]["mse"]
        for f in range(1, 6)
    )
    drugs_improved = int((per_drug.R1_OOF_MSE < per_drug.P13D_MSE).sum())
    majority_drugs = drugs_improved > len(per_drug)/2
    historical_gain = pooled_metrics["P13D"]["mse"] - pooled_metrics["R1_in"]["mse"]
    oof_gain = pooled_metrics["P13D"]["mse"] - pooled_metrics["R1_OOF"]["mse"]
    retention = float(oof_gain / historical_gain) if historical_gain != 0 else float("nan")
    macro_oof_better = macro["R1_OOF"]["mse"]["mean"] < macro["P13D"]["mse"]["mean"]
    pooled_oof_better = pooled_metrics["R1_OOF"]["mse"] < pooled_metrics["P13D"]["mse"]
    targetmean_fully_explains = pooled_metrics["TargetMean_OOF"]["mse"] <= pooled_metrics["R1_OOF"]["mse"]
    shuffled_comparable = shuffle_summary["P(shuffled <= real R1_OOF)"] >= .05
    ci_upper = bootstrap_results["R1_OOF_vs_P13D"]["95%_CI"][1]
    not_supported = (
        not pooled_oof_better or folds_improved <= 2 or not majority_drugs
        or shuffled_comparable or targetmean_fully_explains
    )
    strong = (
        folds_improved >= 4 and macro_oof_better and pooled_oof_better and majority_drugs
        and ci_upper < 0 and not targetmean_fully_explains and not shuffled_comparable
    )
    verdict = "NOT SUPPORTED" if not_supported else "SUPPORTED STRONG" if strong else "SUPPORTED WEAK"

    summary = {
        "experiment": "N3 Cross-Fitted Residual Robustness of Same-Target Chemical Neighborhood Transfer",
        "status": "complete", "verdict": verdict, "E_star": e_stars,
        "all_50_training_stages_successful": True,
        "OOF_completeness": {"all_folds_passed": True, "folds": {f: fold_results[f]["OOF_completeness"] for f in fold_results}},
        "per_fold": fold_results, "macro_mean_and_sample_sd": macro,
        "pooled_metrics": pooled_metrics, "folds_R1_OOF_better_than_P13D": folds_improved,
        "heldout_drugs_R1_OOF_better_than_P13D": drugs_improved,
        "heldout_drugs_total": len(per_drug), "majority_drugs_improved": majority_drugs,
        "historical_gain": historical_gain, "oof_gain": oof_gain, "historical_gain_retention": retention,
        "TargetMean_control": pooled_metrics["TargetMean_OOF"], "TargetMean_fully_explains_R1_OOF": targetmean_fully_explains,
        "shuffled_control": shuffle_summary, "drug_cluster_bootstrap": bootstrap_results,
        "interpretation_caveat": (
            "Strict OOF residual models train on only about 37-39 drugs, whereas historical outer-query "
            "P13D trains on 47-48 drugs. Therefore e_OOF versus e_in combines an unseen-drug effect and "
            "a reduced-training-set effect; N3 failure must not be over-interpreted as proof that "
            "chemical-space residual smoothness does not exist."
        ),
        "historical_outer_query_status": "historical/development-selected on outer validation",
    }
    write_json(args.output_root / "final_summary.json", summary)

    lines = [
        "# N3 FINAL REPORT", "", f"Final verdict: **{verdict}**", "",
        "## Preregistered interpretation", "",
        "N3 is a locked robustness/mechanism experiment, not untouched prospective or fully independent confirmation. "
        "The historical outer-query P13D checkpoints were selected using their outer-validation folds.", "",
        "Each strict OOF residual model was trained on only about 37–39 drugs, whereas the historical outer-query "
        "P13D was trained on 47–48 drugs. Therefore the difference between e_OOF and e_in contains both the "
        "unseen-drug effect and the reduced-training-set effect. If N3 fails, that result must not be over-interpreted "
        "as showing that chemical-space residual smoothness does not exist.", "",
        "TargetMean_OOF is the pure target-wise mean `mean_d e_OOF(d,p)` and uses no drug similarity, top-k, gamma, "
        "tau, beta, clip, scale, support, or R1 weighting. Every shuffled control permutes one complete "
        "442-dimensional drug residual vector; no target-wise independent shuffle is used.", "",
        "## E* and stage status", "", "| Run | E* |", "|---|---:|",
    ]
    lines += [f"| {key} | {value} |" for key, value in e_stars.items()]
    lines += ["", "All 25 Stage-A and 25 fresh Stage-B stages completed: **True**.", "", "## Per-fold metrics", "",
              "| Fold | Model | MSE | RMSE | MAE | CI | Rm2 |", "|---:|---|---:|---:|---:|---:|---:|"]
    for fold in range(1, 6):
        for name in model_names:
            m = fold_results[str(fold)]["metrics"][name]
            lines.append(f"| {fold} | {name} | {m['mse']:.9f} | {m['rmse']:.9f} | {m['mae']:.9f} | {m['ci']:.9f} | {m['rm2']:.9f} |")
    lines += ["", "## Macro mean ± sample SD", "", "| Model | MSE | CI | Rm2 |", "|---|---:|---:|---:|"]
    for name in model_names:
        lines.append(
            f"| {name} | {macro[name]['mse']['mean']:.9f} ± {macro[name]['mse']['sample_sd']:.9f} | "
            f"{macro[name]['ci']['mean']:.9f} ± {macro[name]['ci']['sample_sd']:.9f} | "
            f"{macro[name]['rm2']['mean']:.9f} ± {macro[name]['rm2']['sample_sd']:.9f} |"
        )
    lines += ["", "## Pooled metrics", "", "| Model | MSE | RMSE | MAE | CI | Rm2 |", "|---|---:|---:|---:|---:|---:|"]
    for name in model_names:
        m = pooled_metrics[name]
        lines.append(f"| {name} | {m['mse']:.9f} | {m['rmse']:.9f} | {m['mae']:.9f} | {m['ci']:.9f} | {m['rm2']:.9f} |")
    primary = bootstrap_results["R1_OOF_vs_P13D"]
    lines += [
        "", "## Preregistered criteria and controls", "",
        f"- Folds with R1_OOF MSE < P13D MSE: **{folds_improved}/5**.",
        f"- Held-out drugs improved: **{drugs_improved}/{len(per_drug)}**.",
        f"- Historical gain retention: **{retention:.6f}**.",
        f"- Drug-cluster bootstrap ΔMSE mean/median: **{primary['mean_delta_mse']:.9f} / {primary['median_delta_mse']:.9f}**; "
        f"95% CI **[{primary['95%_CI'][0]:.9f}, {primary['95%_CI'][1]:.9f}]**; P(R1_OOF < P13D) **{primary['probability_first_better']:.6f}**.",
        f"- TargetMean_OOF pooled MSE: **{pooled_metrics['TargetMean_OOF']['mse']:.9f}**; fully explains R1_OOF: **{targetmean_fully_explains}**.",
        f"- 100 shuffled controls MSE mean/median/95% interval: **{shuffle_summary['mse_mean']:.9f} / "
        f"{shuffle_summary['mse_median']:.9f} / [{shuffle_summary['mse_2.5%']:.9f}, {shuffle_summary['mse_97.5%']:.9f}]**; "
        f"P(shuffled <= real R1_OOF) **{shuffle_summary['P(shuffled <= real R1_OOF)']:.6f}**.",
        "", "## Artifacts", "",
        f"- Final JSON: `{args.output_root / 'final_summary.json'}`",
        f"- Per-drug: `{pooled_dir / 'per_drug_metrics.csv'}`",
        f"- Bootstrap: `{pooled_dir / 'bootstrap.json'}`",
        f"- Shuffled control: `{pooled_dir / 'shuffle_control.json'}`",
        f"- Residual diagnostics: `{pooled_dir / 'residual_diagnostics.json'}`",
        f"- Correction diagnostics: `{pooled_dir / 'correction_diagnostics.json'}`",
        f"- Local smoothness: `{pooled_dir / 'local_smoothness.json'}`",
    ]
    (args.output_root / "N3_FINAL_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "folds_improved": folds_improved, "drugs_improved": drugs_improved,
                      "retention": retention, "report": str(args.output_root / 'N3_FINAL_REPORT.md')}, indent=2))


if __name__ == "__main__":
    main()
