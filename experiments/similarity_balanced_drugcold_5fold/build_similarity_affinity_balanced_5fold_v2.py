#!/usr/bin/env python3
"""Build a Davis drug-cold split balanced on chemistry and affinity profiles.

The outer folds partition all 68 drugs exactly once. Inner validation sets are
jointly assigned, contain seven drugs per fold, and are globally unique. The
splitter uses affinity labels only for regression stratification; it never uses
model predictions, checkpoints, or experimental outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SIM_NAMES = [
    "top1_mean", "top1_std", "top1_q25", "top1_median", "top1_q75",
    "top3_mean", "top5_mean", "frac_lt_0.2", "frac_0.2_0.4",
    "frac_0.4_0.6", "frac_ge_0.6",
]
SIM_SCALES = np.asarray(
    [0.08, 0.06, 0.08, 0.08, 0.08, 0.08, 0.08, 0.15, 0.15, 0.15, 0.08],
    dtype=np.float64,
)

LABEL_NAMES = [
    "label_mean", "label_std", "mean_drug_q75", "mean_drug_q90", "mean_drug_q95",
    "frac_ge_6", "frac_ge_7", "frac_ge_8", "drug_mean_std",
    "mean_shift_to_train", "train_mean_constant_mse",
]
LABEL_SCALES = np.asarray(
    [0.10, 0.12, 0.12, 0.16, 0.18, 0.04, 0.025, 0.015, 0.16, 0.10, 0.18],
    dtype=np.float64,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def similarity_profile(query: np.ndarray, reference: np.ndarray, similarity: np.ndarray):
    values = similarity[np.ix_(query, reference)]
    ordered = np.sort(values, axis=1)[:, ::-1]
    top1 = ordered[:, 0]
    top3 = ordered[:, :3].mean(axis=1)
    top5 = ordered[:, :5].mean(axis=1)
    profile = np.asarray([
        top1.mean(), top1.std(), np.quantile(top1, 0.25), np.median(top1),
        np.quantile(top1, 0.75), top3.mean(), top5.mean(),
        np.mean(top1 < 0.2), np.mean((top1 >= 0.2) & (top1 < 0.4)),
        np.mean((top1 >= 0.4) & (top1 < 0.6)), np.mean(top1 >= 0.6),
    ], dtype=np.float64)
    return profile, top1, top3, top5


def build_label_statistics(labels: np.ndarray):
    """Precompute additive/per-drug summaries used during split search."""
    return np.stack([
        labels.mean(axis=1),
        np.mean(labels ** 2, axis=1),
        labels.std(axis=1),
        np.quantile(labels, 0.75, axis=1),
        np.quantile(labels, 0.90, axis=1),
        np.quantile(labels, 0.95, axis=1),
        np.mean(labels >= 6.0, axis=1),
        np.mean(labels >= 7.0, axis=1),
        np.mean(labels >= 8.0, axis=1),
    ], axis=1)


def label_profile(query: np.ndarray, reference: np.ndarray, stats: np.ndarray):
    query_stats = stats[query]
    query_mean = float(query_stats[:, 0].mean())
    query_second_moment = float(query_stats[:, 1].mean())
    query_std = math.sqrt(max(query_second_moment - query_mean ** 2, 0.0))
    reference_mean = float(stats[reference, 0].mean())
    drug_means = query_stats[:, 0]
    profile = np.asarray([
        query_mean, query_std,
        query_stats[:, 3].mean(), query_stats[:, 4].mean(), query_stats[:, 5].mean(),
        query_stats[:, 6].mean(), query_stats[:, 7].mean(), query_stats[:, 8].mean(),
        drug_means.std(), query_mean - reference_mean,
        query_second_moment - 2.0 * reference_mean * query_mean + reference_mean ** 2,
    ], dtype=np.float64)
    return profile


def profile_dict(names, values):
    return {name: float(value) for name, value in zip(names, values)}


def balance_penalty(matrix: np.ndarray, scales: np.ndarray):
    centered = (matrix - matrix.mean(axis=0, keepdims=True)) / scales
    return float(np.mean(centered ** 2))


def outer_objective(folds, similarity, label_stats):
    universe = np.arange(len(label_stats))
    sim_profiles, label_profiles = [], []
    within_penalty = 0.0
    for fold in folds:
        reference = np.setdiff1d(universe, fold)
        sim, _, _, _ = similarity_profile(fold, reference, similarity)
        lab = label_profile(fold, reference, label_stats)
        sim_profiles.append(sim)
        label_profiles.append(lab)
        within = similarity[np.ix_(fold, fold)]
        upper = within[np.triu_indices(len(fold), k=1)]
        within_penalty += float(np.mean(np.maximum(upper - 0.4, 0.0) ** 2))
    sim_profiles = np.stack(sim_profiles)
    label_profiles = np.stack(label_profiles)
    score = (
        0.65 * balance_penalty(sim_profiles, SIM_SCALES)
        + 1.70 * balance_penalty(label_profiles, LABEL_SCALES)
        + 0.20 * within_penalty / len(folds)
    )
    return score, sim_profiles, label_profiles


def optimize_outer(similarity, label_stats, sizes, seed, restarts, steps):
    rng = np.random.default_rng(seed)
    best = (None, math.inf, None, None)
    n = len(label_stats)
    for _ in range(restarts):
        permutation = rng.permutation(n)
        folds, offset = [], 0
        for size in sizes:
            folds.append(np.sort(permutation[offset:offset + size]))
            offset += size
        score, sim_profiles, label_profiles = outer_objective(folds, similarity, label_stats)
        for step in range(steps):
            left, right = rng.choice(5, 2, replace=False)
            li = int(rng.integers(len(folds[left])))
            ri = int(rng.integers(len(folds[right])))
            candidate = [fold.copy() for fold in folds]
            left_value = int(candidate[left][li])
            right_value = int(candidate[right][ri])
            candidate[left][li] = right_value
            candidate[right][ri] = left_value
            candidate[left].sort()
            candidate[right].sort()
            candidate_score, candidate_sim, candidate_label = outer_objective(
                candidate, similarity, label_stats
            )
            fraction = step / max(steps - 1, 1)
            temperature = 0.035 * (0.00008 / 0.035) ** fraction
            if candidate_score < score or rng.random() < math.exp(
                min(0.0, (score - candidate_score) / max(temperature, 1e-12))
            ):
                folds, score = candidate, candidate_score
                sim_profiles, label_profiles = candidate_sim, candidate_label
            if score < best[1]:
                best = ([fold.copy() for fold in folds], score,
                        sim_profiles.copy(), label_profiles.copy())
    return best


def random_unique_inner(outer_folds, rng, validation_size):
    universe = np.arange(sum(len(fold) for fold in outer_folds))
    used = set()
    validations = []
    for fold_index in rng.permutation(5):
        eligible = np.asarray([
            value for value in universe
            if value not in used and value not in set(outer_folds[fold_index])
        ], dtype=int)
        selected = np.sort(rng.choice(eligible, validation_size, replace=False))
        while len(validations) <= fold_index:
            validations.append(None)
        validations[fold_index] = selected
        used.update(map(int, selected))
    return validations


def inner_joint_objective(validations, outer_folds, similarity, label_stats):
    universe = np.arange(len(label_stats))
    sim_profiles, label_profiles = [], []
    target_sim_profiles, target_label_profiles = [], []
    for validation, test in zip(validations, outer_folds):
        development = np.setdiff1d(universe, test)
        train = np.setdiff1d(development, validation)
        sim, _, _, _ = similarity_profile(validation, train, similarity)
        lab = label_profile(validation, train, label_stats)
        target_sim, _, _, _ = similarity_profile(test, development, similarity)
        target_lab = label_profile(test, development, label_stats)
        sim_profiles.append(sim)
        label_profiles.append(lab)
        target_sim_profiles.append(target_sim)
        target_label_profiles.append(target_lab)
    sim_profiles = np.stack(sim_profiles)
    label_profiles = np.stack(label_profiles)
    target_sim_profiles = np.stack(target_sim_profiles)
    target_label_profiles = np.stack(target_label_profiles)
    sim_match = np.mean(((sim_profiles - target_sim_profiles) / SIM_SCALES) ** 2)
    label_match = np.mean(((label_profiles - target_label_profiles) / LABEL_SCALES) ** 2)
    sim_top1_range_penalty = max(float(np.ptp(sim_profiles[:, 0])) - 0.035, 0.0) ** 2 / 0.01 ** 2
    score = (
        0.90 * float(sim_match)
        + 1.45 * float(label_match)
        + 0.90 * balance_penalty(sim_profiles, SIM_SCALES)
        + 1.10 * balance_penalty(label_profiles, LABEL_SCALES)
        + 1.50 * sim_top1_range_penalty
    )
    return score, sim_profiles, label_profiles


def optimize_inner_joint(outer_folds, similarity, label_stats, validation_size,
                         seed, restarts, steps):
    rng = np.random.default_rng(seed)
    universe = set(range(len(label_stats)))
    best = (None, math.inf, None, None)
    for _ in range(restarts):
        validations = random_unique_inner(outer_folds, rng, validation_size)
        score, sim_profiles, label_profiles = inner_joint_objective(
            validations, outer_folds, similarity, label_stats
        )
        for step in range(steps):
            candidate = [values.copy() for values in validations]
            selected = set(map(int, np.concatenate(candidate)))
            if rng.random() < 0.55:
                fold = int(rng.integers(5))
                unused = np.asarray(sorted(
                    universe - selected - set(map(int, outer_folds[fold]))
                ), dtype=int)
                if not len(unused):
                    continue
                position = int(rng.integers(validation_size))
                candidate[fold][position] = int(rng.choice(unused))
                candidate[fold].sort()
            else:
                left, right = map(int, rng.choice(5, 2, replace=False))
                li = int(rng.integers(validation_size))
                ri = int(rng.integers(validation_size))
                left_value = int(candidate[left][li])
                right_value = int(candidate[right][ri])
                if right_value in set(map(int, outer_folds[left])):
                    continue
                if left_value in set(map(int, outer_folds[right])):
                    continue
                candidate[left][li] = right_value
                candidate[right][ri] = left_value
                candidate[left].sort()
                candidate[right].sort()
            candidate_score, candidate_sim, candidate_label = inner_joint_objective(
                candidate, outer_folds, similarity, label_stats
            )
            fraction = step / max(steps - 1, 1)
            temperature = 0.04 * (0.00008 / 0.04) ** fraction
            if candidate_score < score or rng.random() < math.exp(
                min(0.0, (score - candidate_score) / max(temperature, 1e-12))
            ):
                validations, score = candidate, candidate_score
                sim_profiles, label_profiles = candidate_sim, candidate_label
            if score < best[1]:
                best = ([values.copy() for values in validations], score,
                        sim_profiles.copy(), label_profiles.copy())
    return best


def indices_for(pairs, drug_ids):
    return pairs.index[pairs["drug_id"].isin(set(drug_ids))].astype(int).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--similarity-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-restarts", type=int, default=18)
    parser.add_argument("--outer-steps", type=int, default=4000)
    parser.add_argument("--inner-restarts", type=int, default=18)
    parser.add_argument("--inner-steps", type=int, default=4500)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pairs_csv, dtype={"drug_id": str, "protein_id": str})
    required = {"drug_id", "protein_id", "smiles", "label"}
    if not required.issubset(pairs.columns):
        raise RuntimeError(f"Missing columns: {sorted(required - set(pairs.columns))}")
    drug_table = pairs[["drug_id", "smiles"]].drop_duplicates()
    if drug_table["drug_id"].duplicated().any():
        raise RuntimeError("A drug_id maps to multiple SMILES")
    protein_count = int(pairs["protein_id"].nunique())
    counts = pairs.groupby("drug_id").size()
    if counts.nunique() != 1 or int(counts.iloc[0]) != protein_count:
        raise RuntimeError("Davis grid is incomplete")

    cache = np.load(args.similarity_npz, allow_pickle=True)
    drug_ids = [str(value) for value in cache["drug_ids"].tolist()]
    similarity = np.asarray(cache["drug_similarity"], dtype=np.float64)
    if set(drug_ids) != set(drug_table["drug_id"]):
        raise RuntimeError("Similarity IDs do not match pairs.csv")
    labels = np.stack([
        pairs.loc[pairs["drug_id"] == drug, "label"].to_numpy(dtype=np.float64)
        for drug in drug_ids
    ])
    if labels.shape != (len(drug_ids), protein_count):
        raise RuntimeError(f"Unexpected label matrix shape: {labels.shape}")
    label_stats = build_label_statistics(labels)

    base, remainder = divmod(len(drug_ids), 5)
    sizes = [base + (index < remainder) for index in range(5)]
    outer_folds, outer_score, _, _ = optimize_outer(
        similarity, label_stats, sizes, args.seed, args.outer_restarts, args.outer_steps
    )
    universe = np.arange(len(drug_ids))
    if sorted(np.concatenate(outer_folds).tolist()) != universe.tolist():
        raise RuntimeError("Outer folds are not a complete partition")
    validations, inner_score, _, _ = optimize_inner_joint(
        outer_folds, similarity, label_stats, 7, args.seed + 777,
        args.inner_restarts, args.inner_steps,
    )
    inner_flat = list(map(int, np.concatenate(validations)))
    if len(inner_flat) != len(set(inner_flat)):
        raise RuntimeError("Inner validation drugs are not globally unique")

    fold_audits, assignment_rows = [], []
    for fold_number, (test, validation) in enumerate(zip(outer_folds, validations), 1):
        development = np.setdiff1d(universe, test)
        train = np.setdiff1d(development, validation)
        if set(test) & set(development) or set(train) & set(validation):
            raise RuntimeError("Drug overlap detected")
        if not set(validation).issubset(set(development)):
            raise RuntimeError("Validation contains outer-test drug")
        test_ids = [drug_ids[index] for index in test]
        development_ids = [drug_ids[index] for index in development]
        validation_ids = [drug_ids[index] for index in validation]
        train_ids = [drug_ids[index] for index in train]
        test_sim, test_top1, test_top3, test_top5 = similarity_profile(
            test, development, similarity
        )
        val_sim, val_top1, val_top3, val_top5 = similarity_profile(
            validation, train, similarity
        )
        test_label = label_profile(test, development, label_stats)
        val_label = label_profile(validation, train, label_stats)

        split = {
            "name": "davis_similarity_affinity_balanced_drug_cold_5fold_seed42_v2",
            "fold": fold_number,
            "seed": args.seed,
            "protocol": {
                "outer": "five-fold drug-disjoint; each drug is outer-test exactly once",
                "stage_a": "inner train/validation selects epoch",
                "stage_b": "fresh refit on complete outer development for selected epoch",
                "stage_c": "single frozen outer-test evaluation",
                "labels_used_for_regression_stratification": True,
                "model_outputs_used_for_split": False,
                "similarity": "Morgan radius=2, fpSize=2048, Tanimoto locked cache",
                "disclosure": "Outer labels were used only to balance fold distributions; this is a curated stratified benchmark.",
            },
            "inner_train_drugs": train_ids,
            "inner_validation_drugs": validation_ids,
            "outer_development_drugs": development_ids,
            "outer_test_drugs": test_ids,
            "train_drugs": train_ids,
            "val_drugs": validation_ids,
            "test_drugs": test_ids,
            "train_indices": indices_for(pairs, train_ids),
            "val_indices": indices_for(pairs, validation_ids),
            "refit_indices": indices_for(pairs, development_ids),
            "test_indices": indices_for(pairs, test_ids),
            "counts": {
                "inner_train_drugs": len(train_ids),
                "inner_validation_drugs": len(validation_ids),
                "outer_development_drugs": len(development_ids),
                "outer_test_drugs": len(test_ids),
                "inner_train_pairs": len(train_ids) * protein_count,
                "inner_validation_pairs": len(validation_ids) * protein_count,
                "outer_development_pairs": len(development_ids) * protein_count,
                "outer_test_pairs": len(test_ids) * protein_count,
            },
            "audit_profiles": {
                "outer_test_similarity": profile_dict(SIM_NAMES, test_sim),
                "outer_test_affinity": profile_dict(LABEL_NAMES, test_label),
                "inner_validation_similarity": profile_dict(SIM_NAMES, val_sim),
                "inner_validation_affinity": profile_dict(LABEL_NAMES, val_label),
            },
            "assertions": {
                "drug_sets_disjoint_at_each_stage": True,
                "outer_test_used_for_training": False,
                "outer_test_used_for_epoch_selection": False,
                "model_outputs_used_for_split": False,
                "affinity_used_only_for_stratification": True,
                "inner_validation_drugs_globally_unique": True,
            },
        }
        fold_dir = args.output_dir / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        dump_json(fold_dir / "split.json", split)
        fold_audits.append({
            "fold": fold_number,
            "outer_test_size": len(test),
            "outer_development_size": len(development),
            "inner_validation_size": len(validation),
            "inner_train_size": len(train),
            "outer_test_drugs": test_ids,
            "inner_validation_drugs": validation_ids,
            "outer_similarity": profile_dict(SIM_NAMES, test_sim),
            "outer_affinity": profile_dict(LABEL_NAMES, test_label),
            "inner_similarity": profile_dict(SIM_NAMES, val_sim),
            "inner_affinity": profile_dict(LABEL_NAMES, val_label),
        })
        for role, indices, top1, top3, top5 in (
            ("outer_test", test, test_top1, test_top3, test_top5),
            ("inner_validation", validation, val_top1, val_top3, val_top5),
        ):
            for local, index in enumerate(indices):
                assignment_rows.append({
                    "fold": fold_number,
                    "role": role,
                    "drug_id": drug_ids[int(index)],
                    "label_mean": float(labels[int(index)].mean()),
                    "label_std": float(labels[int(index)].std()),
                    "top1_similarity_to_train": float(top1[local]),
                    "top3_similarity_to_train": float(top3[local]),
                    "top5_similarity_to_train": float(top5[local]),
                })

    def dispersion(role, family, names):
        matrix = np.stack([
            np.asarray([fold[family][name] for name in names]) for fold in fold_audits
        ])
        return {
            "mean": profile_dict(names, matrix.mean(axis=0)),
            "sample_sd": profile_dict(names, matrix.std(axis=0, ddof=1)),
            "range": profile_dict(names, np.ptp(matrix, axis=0)),
        }

    audit = {
        "name": "Davis similarity-and-affinity-balanced drug-cold five-fold audit",
        "version": "v2",
        "seed": args.seed,
        "labels_used_for_regression_stratification": True,
        "model_outputs_used_for_split": False,
        "drug_count": len(drug_ids),
        "protein_count": protein_count,
        "pair_count": len(pairs),
        "outer_fold_sizes": sizes,
        "outer_optimization_objective": outer_score,
        "inner_joint_optimization_objective": inner_score,
        "all_outer_test_drugs_exactly_once": True,
        "all_35_inner_validation_drugs_unique": True,
        "outer_similarity_dispersion": dispersion("outer", "outer_similarity", SIM_NAMES),
        "outer_affinity_dispersion": dispersion("outer", "outer_affinity", LABEL_NAMES),
        "inner_similarity_dispersion": dispersion("inner", "inner_similarity", SIM_NAMES),
        "inner_affinity_dispersion": dispersion("inner", "inner_affinity", LABEL_NAMES),
        "folds": fold_audits,
        "provenance": {
            "pairs_csv": str(args.pairs_csv.resolve()),
            "pairs_csv_sha256": sha256(args.pairs_csv),
            "similarity_npz": str(args.similarity_npz.resolve()),
            "similarity_npz_sha256": sha256(args.similarity_npz),
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": sha256(Path(__file__).resolve()),
        },
    }
    gates = {
        "outer_top1_mean_range_le_0.03": audit["outer_similarity_dispersion"]["range"]["top1_mean"] <= 0.03,
        "outer_label_mean_range_le_0.10": audit["outer_affinity_dispersion"]["range"]["label_mean"] <= 0.10,
        "outer_label_std_range_le_0.12": audit["outer_affinity_dispersion"]["range"]["label_std"] <= 0.12,
        "outer_constant_mse_range_le_0.20": audit["outer_affinity_dispersion"]["range"]["train_mean_constant_mse"] <= 0.20,
        "inner_top1_mean_range_le_0.04": audit["inner_similarity_dispersion"]["range"]["top1_mean"] <= 0.04,
        "inner_label_mean_range_le_0.12": audit["inner_affinity_dispersion"]["range"]["label_mean"] <= 0.12,
        "inner_label_std_range_le_0.14": audit["inner_affinity_dispersion"]["range"]["label_std"] <= 0.14,
        "inner_constant_mse_range_le_0.22": audit["inner_affinity_dispersion"]["range"]["train_mean_constant_mse"] <= 0.22,
        "inner_validation_unique_35_of_35": True,
    }
    audit["acceptance_gates"] = gates
    audit["all_acceptance_gates_passed"] = all(gates.values())
    dump_json(args.output_dir / "audit_summary.json", audit)
    pd.DataFrame(assignment_rows).to_csv(
        args.output_dir / "drug_profiles.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    dump_json(args.output_dir / "manifest.json", {
        "split_files": [str((args.output_dir / f"fold_{fold}/split.json").resolve()) for fold in range(1, 6)],
        "audit": str((args.output_dir / "audit_summary.json").resolve()),
        "training_started": False,
        "all_acceptance_gates_passed": audit["all_acceptance_gates_passed"],
    })

    lines = [
        "# Davis Similarity + Affinity Stratified Drug-Cold Split v2", "",
        "Affinity labels were used only for regression stratification. No model outputs were used.",
        "Outer test labels remain excluded from model fitting and epoch selection.", "",
        "| Fold | Train | Inner val | Outer test | Test top1 | Test const-MSE | Val top1 | Val const-MSE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in fold_audits:
        lines.append(
            f"| {fold['fold']} | {fold['inner_train_size']} | {fold['inner_validation_size']} | "
            f"{fold['outer_test_size']} | {fold['outer_similarity']['top1_mean']:.4f} | "
            f"{fold['outer_affinity']['train_mean_constant_mse']:.4f} | "
            f"{fold['inner_similarity']['top1_mean']:.4f} | "
            f"{fold['inner_affinity']['train_mean_constant_mse']:.4f} |"
        )
    lines.extend(["", "## Acceptance gates", ""])
    for name, passed in gates.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    (args.output_dir / "SPLIT_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "outer_score": outer_score,
        "inner_score": inner_score,
        "gates": gates,
        "all_acceptance_gates_passed": all(gates.values()),
    }, indent=2))
    if not audit["all_acceptance_gates_passed"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
