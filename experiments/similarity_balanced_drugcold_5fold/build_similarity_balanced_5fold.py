#!/usr/bin/env python3
"""Build a label-blind, similarity-balanced Davis drug-cold five-fold split.

The outer test folds partition all drugs exactly once.  Fold construction uses
only a precomputed Morgan/Tanimoto matrix.  For each outer fold, a small inner
validation set is selected from the outer-development drugs so its chemical
difficulty profile resembles the outer-test profile.  Affinity labels are
never read by this script.
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


PROFILE_NAMES = [
    "top1_mean", "top1_std", "top1_q25", "top1_median", "top1_q75",
    "top3_mean", "top5_mean", "frac_lt_0.2", "frac_0.2_0.4",
    "frac_0.4_0.6", "frac_ge_0.6",
]
PROFILE_SCALES = np.asarray(
    [0.20, 0.15, 0.20, 0.20, 0.20, 0.20, 0.20, 0.25, 0.25, 0.25, 0.25],
    dtype=np.float64,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def query_statistics(query: np.ndarray, reference: np.ndarray, similarity: np.ndarray):
    values = similarity[np.ix_(query, reference)]
    ordered = np.sort(values, axis=1)[:, ::-1]
    top1 = ordered[:, 0]
    top3 = ordered[:, : min(3, ordered.shape[1])].mean(axis=1)
    top5 = ordered[:, : min(5, ordered.shape[1])].mean(axis=1)
    profile = np.asarray([
        top1.mean(), top1.std(), np.quantile(top1, 0.25), np.median(top1),
        np.quantile(top1, 0.75), top3.mean(), top5.mean(),
        np.mean(top1 < 0.2), np.mean((top1 >= 0.2) & (top1 < 0.4)),
        np.mean((top1 >= 0.4) & (top1 < 0.6)), np.mean(top1 >= 0.6),
    ], dtype=np.float64)
    return profile, top1, top3, top5


def outer_objective(folds: list[np.ndarray], similarity: np.ndarray):
    universe = np.arange(similarity.shape[0])
    profiles = []
    within_penalty = 0.0
    for fold in folds:
        reference = np.setdiff1d(universe, fold, assume_unique=False)
        profile, _, _, _ = query_statistics(fold, reference, similarity)
        profiles.append(profile)
        within = similarity[np.ix_(fold, fold)]
        upper = within[np.triu_indices(len(fold), k=1)]
        # Similar drugs held out together cease to be training analogues.  This
        # penalty disperses strong analogues while leaving the fold-size and
        # distribution-balance terms dominant.
        within_penalty += float(np.mean(np.maximum(upper - 0.4, 0.0) ** 2))
    profiles = np.stack(profiles)
    centered = (profiles - profiles.mean(axis=0, keepdims=True)) / PROFILE_SCALES
    balance = float(np.mean(centered ** 2))
    return balance + 0.30 * within_penalty / len(folds), profiles


def optimize_outer(similarity: np.ndarray, sizes: list[int], seed: int,
                   restarts: int, steps: int):
    rng = np.random.default_rng(seed)
    best_folds, best_score, best_profiles = None, math.inf, None
    n = similarity.shape[0]
    for restart in range(restarts):
        permutation = rng.permutation(n)
        folds, offset = [], 0
        for size in sizes:
            folds.append(np.sort(permutation[offset: offset + size]))
            offset += size
        score, profiles = outer_objective(folds, similarity)
        start_temp, end_temp = 0.025, 0.0001
        for step in range(steps):
            left, right = rng.choice(len(folds), size=2, replace=False)
            li = int(rng.integers(len(folds[left])))
            ri = int(rng.integers(len(folds[right])))
            candidate = [x.copy() for x in folds]
            candidate[left][li], candidate[right][ri] = (
                candidate[right][ri], candidate[left][li]
            )
            candidate[left].sort(); candidate[right].sort()
            candidate_score, candidate_profiles = outer_objective(candidate, similarity)
            fraction = step / max(steps - 1, 1)
            temperature = start_temp * (end_temp / start_temp) ** fraction
            accept = candidate_score < score or rng.random() < math.exp(
                min(0.0, (score - candidate_score) / max(temperature, 1e-12))
            )
            if accept:
                folds, score, profiles = candidate, candidate_score, candidate_profiles
            if score < best_score:
                best_folds = [x.copy() for x in folds]
                best_score, best_profiles = score, profiles.copy()
    return best_folds, best_score, best_profiles


def inner_objective(validation: np.ndarray, development: np.ndarray,
                    target_profile: np.ndarray, similarity: np.ndarray):
    train = np.setdiff1d(development, validation, assume_unique=False)
    profile, _, _, _ = query_statistics(validation, train, similarity)
    difference = (profile - target_profile) / PROFILE_SCALES
    return float(np.mean(difference ** 2)), profile


def optimize_inner(development: np.ndarray, test: np.ndarray, similarity: np.ndarray,
                   validation_size: int, seed: int, restarts: int, steps: int):
    target_profile, _, _, _ = query_statistics(test, development, similarity)
    rng = np.random.default_rng(seed)
    best_validation, best_score, best_profile = None, math.inf, None
    for _ in range(restarts):
        validation = np.sort(rng.choice(development, size=validation_size, replace=False))
        score, profile = inner_objective(validation, development, target_profile, similarity)
        for _ in range(steps):
            train = np.setdiff1d(development, validation, assume_unique=False)
            remove_position = int(rng.integers(len(validation)))
            addition = int(rng.choice(train))
            candidate = validation.copy()
            candidate[remove_position] = addition
            candidate.sort()
            candidate_score, candidate_profile = inner_objective(
                candidate, development, target_profile, similarity
            )
            if candidate_score < score:
                validation, score, profile = candidate, candidate_score, candidate_profile
            if score < best_score:
                best_validation = validation.copy()
                best_score, best_profile = score, profile.copy()
    return best_validation, best_score, target_profile, best_profile


def profile_dict(profile: np.ndarray):
    return {name: float(value) for name, value in zip(PROFILE_NAMES, profile)}


def indices_for(df: pd.DataFrame, drugs: list[str]):
    return df.index[df["drug_id"].isin(set(drugs))].astype(int).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--similarity-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-restarts", type=int, default=24)
    parser.add_argument("--outer-steps", type=int, default=5000)
    parser.add_argument("--inner-restarts", type=int, default=24)
    parser.add_argument("--inner-steps", type=int, default=2500)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Deliberately exclude the label column at read time.
    pairs = pd.read_csv(args.pairs_csv, usecols=["drug_id", "protein_id", "smiles"], dtype=str)
    drug_smiles = pairs[["drug_id", "smiles"]].drop_duplicates()
    if drug_smiles["drug_id"].duplicated().any():
        raise RuntimeError("A drug_id maps to multiple SMILES")
    protein_count = pairs["protein_id"].nunique()
    counts = pairs.groupby("drug_id").size()
    if counts.nunique() != 1 or int(counts.iloc[0]) != protein_count:
        raise RuntimeError("Davis drug-protein grid is incomplete")

    cache = np.load(args.similarity_npz, allow_pickle=True)
    drug_ids = [str(x) for x in cache["drug_ids"].tolist()]
    similarity = cache["drug_similarity"].astype(np.float64)
    if similarity.shape != (len(drug_ids), len(drug_ids)):
        raise RuntimeError("Drug similarity matrix shape mismatch")
    if set(drug_ids) != set(drug_smiles["drug_id"]):
        raise RuntimeError("Similarity drug IDs do not match pairs.csv")
    if not np.allclose(similarity, similarity.T, atol=1e-7):
        raise RuntimeError("Drug similarity matrix is not symmetric")
    if not np.allclose(np.diag(similarity), 1.0, atol=1e-7):
        raise RuntimeError("Drug similarity diagonal is not one")

    # 68 drugs -> 14,14,14,13,13.  This is standard five-fold outer testing.
    base, remainder = divmod(len(drug_ids), 5)
    sizes = [base + (1 if i < remainder else 0) for i in range(5)]
    outer_folds, outer_score, outer_profiles = optimize_outer(
        similarity, sizes, args.seed, args.outer_restarts, args.outer_steps
    )
    universe = np.arange(len(drug_ids))
    if sorted(np.concatenate(outer_folds).tolist()) != universe.tolist():
        raise RuntimeError("Outer folds do not partition all drugs exactly once")

    audit_folds = []
    assignment_rows = []
    for fold_number, test in enumerate(outer_folds, start=1):
        development = np.setdiff1d(universe, test, assume_unique=False)
        validation_size = 7
        validation, inner_score, test_profile, validation_profile = optimize_inner(
            development, test, similarity, validation_size,
            args.seed + 1000 * fold_number,
            args.inner_restarts, args.inner_steps,
        )
        inner_train = np.setdiff1d(development, validation, assume_unique=False)

        development_ids = [drug_ids[i] for i in development]
        test_ids = [drug_ids[i] for i in test]
        validation_ids = [drug_ids[i] for i in validation]
        inner_train_ids = [drug_ids[i] for i in inner_train]
        if set(development_ids) & set(test_ids):
            raise RuntimeError("Outer development/test overlap")
        if set(inner_train_ids) & set(validation_ids):
            raise RuntimeError("Inner train/validation overlap")
        if set(inner_train_ids) | set(validation_ids) != set(development_ids):
            raise RuntimeError("Inner split does not reconstruct outer development")

        test_profile_exact, test_top1, test_top3, test_top5 = query_statistics(
            test, development, similarity
        )
        val_profile_exact, val_top1, val_top3, val_top5 = query_statistics(
            validation, inner_train, similarity
        )
        fold_dir = args.output_dir / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        split = {
            "name": "davis_similarity_balanced_drug_cold_5fold_seed42_v1",
            "fold": fold_number,
            "seed": args.seed,
            "protocol": {
                "outer": "five-fold drug-disjoint evaluation; each drug is outer-test exactly once",
                "stage_a": "inner_train + inner_validation for epoch selection",
                "stage_b": "fresh refit on all outer_development drugs for selected E*",
                "stage_c": "single frozen-checkpoint evaluation on outer_test",
                "labels_used_for_split": False,
                "similarity": "Morgan radius=2, fpSize=2048, Tanimoto; reused locked cache",
            },
            "inner_train_drugs": inner_train_ids,
            "inner_validation_drugs": validation_ids,
            "outer_development_drugs": development_ids,
            "outer_test_drugs": test_ids,
            "train_drugs": inner_train_ids,
            "val_drugs": validation_ids,
            "test_drugs": test_ids,
            "train_indices": indices_for(pairs, inner_train_ids),
            "val_indices": indices_for(pairs, validation_ids),
            "refit_indices": indices_for(pairs, development_ids),
            "test_indices": indices_for(pairs, test_ids),
            "counts": {
                "inner_train_drugs": len(inner_train_ids),
                "inner_validation_drugs": len(validation_ids),
                "outer_development_drugs": len(development_ids),
                "outer_test_drugs": len(test_ids),
                "inner_train_pairs": len(inner_train_ids) * protein_count,
                "inner_validation_pairs": len(validation_ids) * protein_count,
                "outer_development_pairs": len(development_ids) * protein_count,
                "outer_test_pairs": len(test_ids) * protein_count,
            },
            "similarity_audit": {
                "outer_test_vs_outer_development": profile_dict(test_profile_exact),
                "inner_validation_vs_inner_train": profile_dict(val_profile_exact),
                "inner_profile_match_objective": inner_score,
            },
            "assertions": {
                "drug_sets_disjoint_at_each_stage": True,
                "outer_test_used_for_training": False,
                "outer_test_used_for_epoch_selection": False,
                "labels_used_for_split": False,
                "each_drug_has_pairs_for_all_proteins": True,
            },
        }
        json_dump(fold_dir / "split.json", split)

        for role, indices, top1, top3, top5 in (
            ("outer_test", test, test_top1, test_top3, test_top5),
            ("inner_validation", validation, val_top1, val_top3, val_top5),
        ):
            for local, drug in enumerate(indices):
                assignment_rows.append({
                    "fold": fold_number, "role": role, "drug_id": drug_ids[int(drug)],
                    "top1_similarity_to_corresponding_train": float(top1[local]),
                    "top3_mean_similarity_to_corresponding_train": float(top3[local]),
                    "top5_mean_similarity_to_corresponding_train": float(top5[local]),
                })
        audit_folds.append({
            "fold": fold_number,
            "outer_test_size": len(test_ids),
            "outer_development_size": len(development_ids),
            "inner_train_size": len(inner_train_ids),
            "inner_validation_size": len(validation_ids),
            "outer_profile": profile_dict(test_profile_exact),
            "inner_validation_profile": profile_dict(val_profile_exact),
            "inner_profile_match_objective": inner_score,
            "outer_test_drugs": test_ids,
            "inner_validation_drugs": validation_ids,
        })

    profile_matrix = np.stack([
        np.asarray([fold["outer_profile"][name] for name in PROFILE_NAMES])
        for fold in audit_folds
    ])
    audit = {
        "name": "Davis similarity-balanced drug-cold five-fold split audit",
        "version": "v1",
        "seed": args.seed,
        "labels_read": False,
        "drug_count": len(drug_ids),
        "protein_count": protein_count,
        "pair_count": len(pairs),
        "outer_fold_sizes": sizes,
        "outer_optimization_objective": outer_score,
        "outer_profile_across_fold_mean": profile_dict(profile_matrix.mean(0)),
        "outer_profile_across_fold_sample_sd": profile_dict(profile_matrix.std(0, ddof=1)),
        "outer_profile_across_fold_range": profile_dict(profile_matrix.max(0)-profile_matrix.min(0)),
        "all_outer_test_drugs_exactly_once": True,
        "folds": audit_folds,
        "provenance": {
            "pairs_csv": str(args.pairs_csv.resolve()),
            "pairs_csv_sha256": sha256(args.pairs_csv),
            "similarity_npz": str(args.similarity_npz.resolve()),
            "similarity_npz_sha256": sha256(args.similarity_npz),
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": sha256(Path(__file__).resolve()),
        },
    }
    json_dump(args.output_dir / "audit_summary.json", audit)
    pd.DataFrame(assignment_rows).to_csv(
        args.output_dir / "similarity_profiles.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    json_dump(args.output_dir / "manifest.json", {
        "split_files": [str((args.output_dir / f"fold_{i}/split.json").resolve()) for i in range(1, 6)],
        "audit": str((args.output_dir / "audit_summary.json").resolve()),
        "similarity_profiles": str((args.output_dir / "similarity_profiles.csv").resolve()),
        "training_started": False,
    })

    lines = [
        "# Davis Similarity-Balanced Drug-Cold 5-Fold Split", "",
        "This split was built without reading affinity labels or model outputs.", "",
        "- Every drug is an outer-test drug exactly once.",
        "- Outer folds contain 14/14/14/13/13 drugs.",
        "- Each fold uses seven inner-validation drugs for epoch selection.",
        "- A fresh Stage-B refit should use all outer-development drugs before one test evaluation.",
        "- No training was launched by the split builder.", "",
        "| Fold | Inner train | Inner val | Outer development | Outer test | Test top1 mean | Test top1 median | Test top5 mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in audit_folds:
        p = fold["outer_profile"]
        lines.append(
            f"| {fold['fold']} | {fold['inner_train_size']} | {fold['inner_validation_size']} | "
            f"{fold['outer_development_size']} | {fold['outer_test_size']} | "
            f"{p['top1_mean']:.6f} | {p['top1_median']:.6f} | {p['top5_mean']:.6f} |"
        )
    lines.extend(["", "## Across-fold dispersion", "",
                  "| Statistic | Mean | Sample SD | Range |", "|---|---:|---:|---:|"])
    for name in PROFILE_NAMES:
        lines.append(
            f"| {name} | {audit['outer_profile_across_fold_mean'][name]:.6f} | "
            f"{audit['outer_profile_across_fold_sample_sd'][name]:.6f} | "
            f"{audit['outer_profile_across_fold_range'][name]:.6f} |"
        )
    (args.output_dir / "SPLIT_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "outer_score": outer_score,
        "outer_fold_sizes": sizes,
        "top1_means": [fold["outer_profile"]["top1_mean"] for fold in audit_folds],
        "top1_medians": [fold["outer_profile"]["top1_median"] for fold in audit_folds],
        "top5_means": [fold["outer_profile"]["top5_mean"] for fold in audit_folds],
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
