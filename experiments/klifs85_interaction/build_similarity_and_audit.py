#!/usr/bin/env python3
"""Build Davis drug/pocket similarities and audit the joint-similarity signal.

The held-out test indices are never selected or scored.  The audit asks whether
ECFP4(drug) * KLIFS85(target) similarity predicts local affinity smoothness on
train and transfers to validation drugs in the drug-cold split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else 0.0
    spear = float(spearmanr(y_true, y_pred).statistic) if np.std(y_pred) > 0 else 0.0
    return {"mse": mse, "rmse": rmse, "pearson": pearson, "spearman": spear}


def pocket_similarity(cache_dir: Path, protein_ids):
    audit_path = cache_dir.parent / "mapping_audit.csv"
    accepted = None
    if audit_path.exists():
        audit = pd.read_csv(audit_path, dtype={"protein_id": str})
        accepted = set(
            audit.loc[audit["status"].isin(["ok", "ok_sequence_fallback"]), "protein_id"]
        )
    sequences, masks, coverage = [], [], []
    for protein_id in protein_ids:
        path = cache_dir / f"{protein_id}.pt"
        if not path.exists() or (accepted is not None and protein_id not in accepted):
            sequences.append(np.full(85, "-", dtype="U1"))
            masks.append(np.zeros(85, dtype=bool))
            coverage.append(0.0)
            continue
        item = torch.load(path, map_location="cpu", weights_only=False)
        sequence = np.asarray(list(item["pocket_sequence"]), dtype="U1")
        mask = item["mask"].numpy().astype(bool)
        sequences.append(sequence)
        masks.append(mask)
        coverage.append(float(mask.mean()))
    sequences, masks = np.stack(sequences), np.stack(masks)
    n = len(protein_ids)
    similarity = np.zeros((n, n), dtype=np.float32)
    overlap_count = np.zeros((n, n), dtype=np.int16)
    for i in range(n):
        overlap = masks[i][None, :] & masks
        count = overlap.sum(axis=1)
        matches = ((sequences[i][None, :] == sequences) & overlap).sum(axis=1)
        # Identity on common standardized positions, penalized for missing sites.
        similarity[i] = np.divide(matches, 85.0, out=np.zeros(n), where=count > 0)
        overlap_count[i] = count
    return similarity, overlap_count, np.asarray(coverage, dtype=np.float32)


def drug_similarity(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for text in smiles:
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            raise ValueError(f"Invalid SMILES: {text}")
        fingerprints.append(generator.GetFingerprint(molecule))
    matrix = np.zeros((len(fingerprints), len(fingerprints)), dtype=np.float32)
    for i, fp in enumerate(fingerprints):
        matrix[i] = DataStructs.BulkTanimotoSimilarity(fp, fingerprints)
    return fingerprints, matrix


def top_joint_neighbors(d, p, allowed_drugs, drug_sim, protein_sim, top_d=16, top_p=64, top_k=24):
    ds = allowed_drugs[np.argsort(drug_sim[d, allowed_drugs])[::-1][:top_d]]
    ps = np.argsort(protein_sim[p])[::-1][:top_p]
    scores = drug_sim[d, ds][:, None] * protein_sim[p, ps][None, :]
    flat = scores.ravel()
    order = np.argsort(flat)[::-1]
    result = []
    for index in order:
        di, pi = np.unravel_index(index, scores.shape)
        nd, np_ = int(ds[di]), int(ps[pi])
        if nd == d and np_ == p:
            continue
        result.append((nd, np_, float(flat[index])))
        if len(result) == top_k:
            break
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--klifs-cache", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=24)
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(root / "data/raw/davis/pairs.csv", dtype={"drug_id": str, "protein_id": str})
    split = json.loads(args.split_json.read_text())
    train_indices = np.asarray(split["train_indices"], dtype=int)
    val_indices = np.asarray(split["val_indices"], dtype=int)
    # Deliberately do not read split["test_indices"] or compute test metrics.
    allowed_indices = np.concatenate([train_indices, val_indices])
    audit_pairs = pairs.iloc[allowed_indices].copy()

    drug_table = pairs[["drug_id", "smiles"]].drop_duplicates("drug_id").reset_index(drop=True)
    protein_table = pairs[["protein_id"]].drop_duplicates("protein_id").reset_index(drop=True)
    drug_ids = drug_table["drug_id"].tolist()
    protein_ids = protein_table["protein_id"].tolist()
    drug_to_index = {name: i for i, name in enumerate(drug_ids)}
    protein_to_index = {name: i for i, name in enumerate(protein_ids)}

    _, drug_sim = drug_similarity(drug_table["smiles"].tolist())
    protein_sim, protein_overlap, protein_coverage = pocket_similarity(args.klifs_cache, protein_ids)
    np.savez_compressed(
        output / "entity_similarities.npz",
        drug_similarity=drug_sim,
        protein_similarity=protein_sim,
        protein_overlap=protein_overlap,
        protein_coverage=protein_coverage,
        drug_ids=np.asarray(drug_ids),
        protein_ids=np.asarray(protein_ids),
    )

    train = pairs.iloc[train_indices].copy()
    val = pairs.iloc[val_indices].copy()
    train_drug_indices = np.asarray([drug_to_index[x] for x in split["train_drugs"]], dtype=int)
    label_grid = np.full((len(drug_ids), len(protein_ids)), np.nan, dtype=np.float32)
    for row in train.itertuples(index=False):
        label_grid[drug_to_index[row.drug_id], protein_to_index[row.protein_id]] = float(row.label)

    edge_rows = []
    for row in train.itertuples(index=False):
        d, p, label = drug_to_index[row.drug_id], protein_to_index[row.protein_id], float(row.label)
        for nd, np_, score in top_joint_neighbors(
            d, p, train_drug_indices, drug_sim, protein_sim, top_k=args.top_k
        ):
            neighbor_label = label_grid[nd, np_]
            if np.isnan(neighbor_label):
                continue
            edge_rows.append((d, p, nd, np_, score, abs(label - float(neighbor_label))))
    edges = pd.DataFrame(
        edge_rows,
        columns=["drug_index", "protein_index", "neighbor_drug_index", "neighbor_protein_index", "joint_similarity", "absolute_label_difference"],
    )
    edges.to_csv(output / "train_joint_neighbor_audit.csv", index=False)

    similarities = edges["joint_similarity"].to_numpy()
    differences = edges["absolute_label_difference"].to_numpy()
    smoothness = float(spearmanr(similarities, differences).statistic)
    bins = pd.qcut(edges["joint_similarity"], q=10, duplicates="drop")
    deciles = edges.groupby(bins, observed=True).agg(
        n=("joint_similarity", "size"),
        similarity_mean=("joint_similarity", "mean"),
        affinity_difference_mean=("absolute_label_difference", "mean"),
        affinity_difference_median=("absolute_label_difference", "median"),
    )
    deciles.to_csv(output / "train_similarity_deciles.csv")

    train_mean = float(train["label"].mean())
    joint_predictions, drug_predictions = [], []
    joint_support = []
    for row in val.itertuples(index=False):
        d, p = drug_to_index[row.drug_id], protein_to_index[row.protein_id]
        neighbors = top_joint_neighbors(
            d, p, train_drug_indices, drug_sim, protein_sim, top_d=len(train_drug_indices), top_p=96, top_k=64
        )
        values = [(score, label_grid[nd, np_]) for nd, np_, score in neighbors if not np.isnan(label_grid[nd, np_])]
        weights = np.asarray([score for score, _ in values], dtype=float)
        labels = np.asarray([label for _, label in values], dtype=float)
        joint_predictions.append(float(np.average(labels, weights=weights)) if weights.sum() > 1e-8 else train_mean)
        joint_support.append(float(weights.sum()))

        weights = drug_sim[d, train_drug_indices].astype(float)
        labels = label_grid[train_drug_indices, p].astype(float)
        drug_predictions.append(float(np.average(labels, weights=weights)) if weights.sum() > 1e-8 else train_mean)

    val_output = val[["drug_id", "protein_id", "label"]].copy()
    val_output["global_mean_prediction"] = train_mean
    val_output["drug_only_kernel_prediction"] = drug_predictions
    val_output["joint_kernel_prediction"] = joint_predictions
    val_output["joint_support"] = joint_support
    val_output.to_csv(output / "validation_kernel_predictions.csv", index=False)

    metrics = {
        "guardrail": "test indices and test metrics were not accessed",
        "n_train": len(train),
        "n_validation": len(val),
        "n_audit_edges": len(edges),
        "n_proteins_with_klifs": int((protein_coverage > 0).sum()),
        "protein_coverage_mean": float(protein_coverage.mean()),
        "train_similarity_vs_abs_affinity_difference_spearman": smoothness,
        "high_similarity_cliff_rate": float(np.mean(differences[similarities >= np.quantile(similarities, 0.9)] >= 2.0)),
        "validation": {
            "global_mean": regression_metrics(val["label"], np.full(len(val), train_mean)),
            "drug_only_kernel": regression_metrics(val["label"], drug_predictions),
            "joint_kernel": regression_metrics(val["label"], joint_predictions),
        },
    }
    (output / "audit_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
