#!/usr/bin/env python3
"""Build label-free drug and protein similarity matrices for Davis pairs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def quantiles(matrix):
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool)
    values = matrix[mask].numpy()
    return {str(q): float(np.quantile(values, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--global-cache", type=Path, required=True)
    parser.add_argument("--protein-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs)
    pairs["drug_id"] = pairs["drug_id"].astype(str)
    pairs["protein_id"] = pairs["protein_id"].astype(str)
    smiles = pairs.drop_duplicates("drug_id").set_index("drug_id")["smiles"].to_dict()
    global_data = torch.load(args.global_cache, map_location="cpu", weights_only=False)
    drug_ids = sorted({str(x) for x in global_data["drug_id"]})
    protein_ids = sorted({str(x) for x in global_data["protein_id"]})

    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for drug_id in drug_ids:
        mol = Chem.MolFromSmiles(smiles[drug_id])
        if mol is None:
            raise ValueError(f"invalid SMILES for {drug_id}")
        fps.append(fp_generator.GetFingerprint(mol))
    drug_similarity = torch.eye(len(drug_ids), dtype=torch.float32)
    for i in range(len(drug_ids)):
        for j in range(i):
            value = float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            drug_similarity[i, j] = drug_similarity[j, i] = value

    protein_embeddings = []
    for protein_id in protein_ids:
        obj = torch.load(args.protein_dir / f"{protein_id}.pt", map_location="cpu", weights_only=False)
        protein_embeddings.append(obj["mean"].float())
    protein_embeddings = F.normalize(torch.stack(protein_embeddings), dim=-1)
    protein_similarity = protein_embeddings @ protein_embeddings.T
    protein_similarity = protein_similarity.clamp(-1.0, 1.0)

    payload = {
        "drug_ids": drug_ids,
        "protein_ids": protein_ids,
        "drug_similarity": drug_similarity,
        "protein_similarity": protein_similarity,
        "metadata": {
            "drugs": len(drug_ids), "proteins": len(protein_ids),
            "drug_similarity": "Morgan radius=2 fpSize=2048 Tanimoto",
            "protein_similarity": "cosine similarity of frozen ESM2 mean embeddings",
            "guardrail": "molecular structures and pretrained protein embeddings only; no affinity labels",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    audit = {
        **payload["metadata"],
        "drug_off_diagonal_quantiles": quantiles(drug_similarity),
        "protein_off_diagonal_quantiles": quantiles(protein_similarity),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()

