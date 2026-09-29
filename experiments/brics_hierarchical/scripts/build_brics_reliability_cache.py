#!/usr/bin/env python3
"""Build label-free drug-support and BRICS-topology reliability features."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--global-cache", type=Path, required=True)
    parser.add_argument("--drug-cache", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs)
    pairs["drug_id"] = pairs["drug_id"].astype(str)
    smiles_by_id = pairs.drop_duplicates("drug_id").set_index("drug_id")["smiles"].to_dict()
    global_data = torch.load(args.global_cache, map_location="cpu", weights_only=False)
    graph_data = torch.load(args.drug_cache, map_location="cpu", weights_only=False)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    row_drugs = [str(x) for x in global_data["drug_id"]]
    train_ids = sorted({row_drugs[i] for i in split["train_indices"]})
    drug_ids = sorted(graph_data["drugs"])

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = {}
    for drug_id in drug_ids:
        mol = Chem.MolFromSmiles(smiles_by_id[drug_id])
        if mol is None:
            raise ValueError(f"invalid SMILES for {drug_id}")
        fingerprints[drug_id] = generator.GetFingerprint(mol)

    raw = {}
    for drug_id in drug_ids:
        candidates = [x for x in train_ids if x != drug_id]
        similarities = sorted(
            (float(DataStructs.TanimotoSimilarity(fingerprints[drug_id], fingerprints[x])) for x in candidates),
            reverse=True,
        )
        if not similarities:
            raise RuntimeError(f"no reference training drugs for {drug_id}")
        item = graph_data["drugs"][drug_id]
        n_fragments = int(item["n_fragments"])
        edge_index = item["fragment_edge_index"].long()
        nonself = edge_index[:, edge_index[0] != edge_index[1]]
        undirected_edges = int(nonself.shape[1] // 2)
        degree = torch.zeros(n_fragments, dtype=torch.float32)
        if nonself.numel():
            degree.index_add_(0, nonself[0], torch.ones(nonself.shape[1]))
        density = 0.0 if n_fragments < 2 else undirected_edges / (n_fragments * (n_fragments - 1) / 2)
        raw[drug_id] = np.asarray([
            similarities[0],
            float(np.mean(similarities[: min(3, len(similarities))])),
            float(np.mean(similarities[: min(5, len(similarities))])),
            float(np.mean(np.asarray(similarities) >= 0.5)),
            float(np.log1p(n_fragments)),
            float(density),
            float(degree.mean()) if n_fragments else 0.0,
            float((degree >= 3).float().mean()) if n_fragments else 0.0,
        ], dtype=np.float32)

    train_matrix = np.stack([raw[x] for x in train_ids])
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    features = {x: torch.from_numpy((raw[x] - mean) / std).float() for x in drug_ids}
    payload = {
        "features": features,
        "raw_features": {x: torch.from_numpy(raw[x]) for x in drug_ids},
        "train_drug_ids": train_ids,
        "feature_names": [
            "max_train_tanimoto", "top3_train_tanimoto", "top5_train_tanimoto",
            "fraction_train_tanimoto_ge_0.5", "log1p_fragments", "fragment_edge_density",
            "mean_fragment_degree", "fraction_fragment_degree_ge_3",
        ],
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "metadata": {
            "drugs": len(drug_ids), "train_drugs": len(train_ids),
            "fingerprint": "Morgan radius=2 fpSize=2048",
            "guardrail": "features use molecular structure and train split membership only; no affinity labels",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    audit = {
        **payload["metadata"], "feature_names": payload["feature_names"],
        "normalization": payload["normalization"], "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()

