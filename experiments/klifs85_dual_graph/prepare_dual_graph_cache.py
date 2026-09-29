#!/usr/bin/env python3
"""Cache frozen P13D node features as BRICS fragment and KLIFS-85 pocket graphs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import BRICS


def bond_one_hot(bond):
    bond_type = bond.GetBondType()
    types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    return [float(bond_type == item) for item in types]


def brics_label_value(label):
    digits = "".join(character for character in str(label) if character.isdigit())
    return float(digits or 0) / 16.0


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def fragment_graph(smiles, atom_features, positions):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit failed to parse SMILES")
    # The existing 3D graph cache contains explicit hydrogens. RDKit appends
    # them after heavy atoms, preserving the SMILES heavy-atom indices used by
    # BRICS and assigning each hydrogen to its bonded fragment.
    mol = Chem.AddHs(mol)
    if mol.GetNumAtoms() != atom_features.shape[0]:
        raise ValueError(f"atom count mismatch RDKit={mol.GetNumAtoms()} graph={atom_features.shape[0]}")
    cuts = list(BRICS.FindBRICSBonds(mol))
    cut_pairs = {tuple(sorted(pair)) for pair, _ in cuts}
    cut_labels = {tuple(sorted(pair)): labels for pair, labels in cuts}
    union = UnionFind(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        pair = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        if pair not in cut_pairs:
            union.union(*pair)
    roots = [union.find(i) for i in range(mol.GetNumAtoms())]
    unique_roots = {root: index for index, root in enumerate(sorted(set(roots)))}
    assignment = torch.tensor([unique_roots[root] for root in roots], dtype=torch.long)
    n_fragments = len(unique_roots)
    fragment_features, centroids, atom_counts = [], [], []
    for fragment in range(n_fragments):
        mask = assignment == fragment
        values = atom_features[mask]
        fragment_features.append(torch.cat([values.mean(0), values.max(0).values], dim=0))
        centroids.append(positions[mask].mean(0))
        atom_counts.append(int(mask.sum()))
    fragment_features = torch.stack(fragment_features)
    centroids = torch.stack(centroids)
    edges, attributes = [], []
    for (a, b), labels in cuts:
        fa, fb = int(assignment[a]), int(assignment[b])
        if fa == fb:
            raise RuntimeError("BRICS cut did not separate components")
        bond = mol.GetBondBetweenAtoms(a, b)
        distance = float(torch.linalg.vector_norm(centroids[fa] - centroids[fb])) / 10.0
        la, lb = brics_label_value(labels[0]), brics_label_value(labels[1])
        base = bond_one_hot(bond)
        edges.extend([(fa, fb), (fb, fa)])
        attributes.extend([base + [la, lb, distance], base + [lb, la, distance]])
    for fragment in range(n_fragments):
        edges.append((fragment, fragment))
        attributes.append([0.0] * 7)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attributes, dtype=torch.float32)
    return {
        "x": fragment_features.cpu(),
        "pos": centroids.cpu(),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "atom_to_fragment": assignment,
        "atom_counts": torch.tensor(atom_counts, dtype=torch.long),
        "n_brics_cuts": len(cuts),
    }


def pocket_graph(node_features, sequence_features, coords, sequence_indices, valid):
    n_positions = 85
    x = torch.zeros((n_positions, node_features.shape[1] + sequence_features.shape[1]), dtype=torch.float32)
    x[valid, : node_features.shape[1]] = node_features[sequence_indices[valid]].float().cpu()
    x[valid, node_features.shape[1] :] = sequence_features[valid].float().cpu()
    valid_ids = torch.where(valid)[0].tolist()
    edge_set = set()
    if len(valid_ids) > 1:
        valid_coords = coords[valid].float()
        distances = torch.cdist(valid_coords, valid_coords)
        for local_i, source in enumerate(valid_ids):
            order = torch.argsort(distances[local_i])
            for local_j in order[1 : min(9, len(order))]:
                target = valid_ids[int(local_j)]
                edge_set.add((source, target))
                edge_set.add((target, source))
    for i in valid_ids:
        edge_set.add((i, i))
    for a in valid_ids:
        for b in valid_ids:
            if a < b and abs(int(sequence_indices[a]) - int(sequence_indices[b])) == 1:
                edge_set.add((a, b))
                edge_set.add((b, a))
    centers = torch.linspace(0.0, 2.0, 16)
    width = float(centers[1] - centers[0])
    edges, attrs = [], []
    for source, target in sorted(edge_set):
        distance_angstrom = float(torch.linalg.vector_norm(coords[source] - coords[target]))
        scaled = distance_angstrom / 10.0
        rbf = torch.exp(-((torch.tensor(scaled) - centers) / width) ** 2).tolist()
        seq_gap = abs(int(sequence_indices[source]) - int(sequence_indices[target]))
        backbone = float(source != target and seq_gap == 1)
        attrs.append(rbf + [backbone, min(seq_gap, 100) / 100.0, abs(source - target) / 84.0])
        edges.append((source, target))
    if not edges:
        edges, attrs = [(0, 0)], [[0.0] * 19]
    return {
        "x": x,
        "mask": valid.cpu(),
        "coords": coords.float().cpu(),
        "sequence_indices": sequence_indices.cpu(),
        "klifs_index": torch.arange(85, dtype=torch.long),
        "edge_index": torch.tensor(edges, dtype=torch.long).t().contiguous(),
        "edge_attr": torch.tensor(attrs, dtype=torch.float32),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/data1/ztx/MyModel-MDTA"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--klifs-cache", type=Path, required=True)
    parser.add_argument("--klifs-mapping-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    root = args.project.resolve()
    sys.path.insert(0, str(root))
    from models.model_p13d import MyModelMDTAP13D

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    model = MyModelMDTAP13D(
        drug_1d_in_dim=saved["drug_1d_in_dim"],
        drug_3d_node_in_dim=saved["drug_3d_node_in_dim"],
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=saved["hidden_dim"],
        dropout=saved["dropout"],
        task="regression",
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    pairs = pd.read_csv(root / "data/raw/davis/pairs.csv", dtype={"drug_id": str, "protein_id": str})
    drug_rows = pairs.drop_duplicates("drug_id").set_index("drug_id")
    protein_ids = pairs["protein_id"].drop_duplicates().tolist()
    klifs = torch.load(args.klifs_cache, map_location="cpu", weights_only=False)["proteins"]

    drugs, drug_audit = {}, []
    for drug_id, row in drug_rows.iterrows():
        raw = torch.load(root / "data/processed/davis/drug_3d" / f"{drug_id}.pt", map_location="cpu", weights_only=False)
        batch = {
            "x": raw["x"].float().to(args.device),
            "pos": raw["pos"].float().to(args.device),
            "edge_index": raw["edge_index"].long().to(args.device),
            "batch": torch.zeros(len(raw["x"]), dtype=torch.long, device=args.device),
        }
        encoded = model.drug_3d_encoder(batch, return_node=True)["node_feat"].cpu()
        graph = fragment_graph(row["smiles"], encoded, raw["pos"].float())
        drugs[str(drug_id)] = graph
        drug_audit.append({
            "drug_id": str(drug_id), "atoms": len(raw["x"]), "fragments": len(graph["x"]),
            "brics_cuts": graph["n_brics_cuts"], "fragment_edges": graph["edge_index"].shape[1],
        })

    proteins, protein_audit = {}, []
    for protein_id in protein_ids:
        raw = torch.load(root / "data/processed/davis/protein_3d_gvp" / f"{protein_id}.pt", map_location="cpu", weights_only=False)
        batch = {
            key: raw[key].to(args.device) for key in ("node_s", "node_v", "coords", "edge_index", "edge_s", "edge_v")
        }
        batch["batch"] = torch.zeros(len(raw["node_s"]), dtype=torch.long, device=args.device)
        encoded = model.protein_3d_encoder(batch, return_node=True)["node_feat"].cpu()
        item = klifs[str(protein_id)]
        mapping_path = args.klifs_mapping_dir / "by_protein" / f"{protein_id}.pt"
        if mapping_path.exists():
            mapping = torch.load(mapping_path, map_location="cpu", weights_only=False)
            indices = mapping["sequence_indices"].long()
        else:
            indices = torch.full((85,), -1, dtype=torch.long)
        valid = item["mask"].bool() & (indices >= 0) & (indices < len(encoded))
        coords = item.get("coords", torch.zeros((85, 3))).float()
        graph = pocket_graph(encoded, item["sequence"], coords, indices, valid)
        proteins[str(protein_id)] = graph
        protein_audit.append({
            "protein_id": str(protein_id), "residues": len(encoded), "valid_klifs": int(valid.sum()),
            "accepted": bool(valid.any()), "pocket_edges": graph["edge_index"].shape[1],
            "mapping_status": item.get("mapping_status", "unknown"),
        })

    payload = {
        "drugs": drugs,
        "proteins": proteins,
        "metadata": {
            "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"),
            "drug_node_dim": 128, "fragment_node_dim": 256, "fragment_edge_dim": 7,
            "pocket_node_dim": 1152, "pocket_edge_dim": 19,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    audit = {
        "drugs": len(drugs), "proteins": len(proteins),
        "accepted_proteins": sum(row["accepted"] for row in protein_audit),
        "drug_fragment_count": {
            "min": min(row["fragments"] for row in drug_audit),
            "mean": float(np.mean([row["fragments"] for row in drug_audit])),
            "max": max(row["fragments"] for row in drug_audit),
        },
        "valid_klifs_count": {
            "min": min(row["valid_klifs"] for row in protein_audit),
            "mean": float(np.mean([row["valid_klifs"] for row in protein_audit])),
            "max": max(row["valid_klifs"] for row in protein_audit),
        },
        "checkpoint_epoch": checkpoint.get("epoch"),
    }
    (args.output.parent / "cache_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    pd.DataFrame(drug_audit).to_csv(args.output.parent / "drug_fragment_audit.csv", index=False)
    pd.DataFrame(protein_audit).to_csv(args.output.parent / "protein_pocket_audit.csv", index=False)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
