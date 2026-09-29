#!/usr/bin/env python3
"""Build scaffold-disjoint and exact-protein-disjoint splits for aligned PLIP pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def parse_sequences(path: Path):
    lines = [line.strip() for line in path.open() if line.strip()]
    return {lines[i][1:].lower(): lines[i + 1] for i in range(0, len(lines), 3)}


def scaffold_key(sdf: Path):
    mol = Chem.MolFromMolFile(str(sdf), removeHs=True, sanitize=True)
    if mol is None:
        raise ValueError("RDKit failed to parse ligand")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    smiles = Chem.MolToSmiles(scaffold, canonical=True)
    return smiles or f"ACYCLIC:{Chem.MolToSmiles(mol, canonical=True)}"


def grouped_split(ids, keys, seed):
    groups = defaultdict(list)
    for pdb_id in ids:
        groups[keys[pdb_id]].append(pdb_id)
    rng = random.Random(seed)
    grouped = list(groups.values())
    rng.shuffle(grouped)
    grouped.sort(key=len, reverse=True)
    targets = {"train": 0.8 * len(ids), "val": 0.1 * len(ids), "test": 0.1 * len(ids)}
    splits = {name: [] for name in targets}
    for group in grouped:
        destination = min(
            splits,
            key=lambda name: len(splits[name]) / max(targets[name], 1),
        )
        splits[destination].extend(group)
    return {name: sorted(values) for name, values in splits.items()}


def verify(splits, keys):
    key_sets = {
        name: {keys[pdb_id] for pdb_id in ids} for name, ids in splits.items()
    }
    return {
        "train_val_overlap": len(key_sets["train"] & key_sets["val"]),
        "train_test_overlap": len(key_sets["train"] & key_sets["test"]),
        "val_test_overlap": len(key_sets["val"] & key_sets["test"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--alignment-audit", type=Path, required=True)
    parser.add_argument("--site-labels", type=Path, required=True)
    parser.add_argument("--refined-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = json.loads(args.pairs.read_text())
    audit = json.loads(args.alignment_audit.read_text())
    aligned = {
        row["pdb_id"] for row in audit["rows"]
        if row["atom_count_plip"] == row["atom_count_graph"]
        and row["residue_count_plip"] == row["residue_count_graph"]
    }
    records = {
        record["pdb_id"]: record for record in pairs["records"]
        if record["pdb_id"] in aligned and record["pairs"]
    }
    sequences = parse_sequences(args.site_labels)
    valid_ids, failures, scaffold_keys, protein_keys = [], [], {}, {}
    for pdb_id in sorted(records):
        try:
            scaffold_keys[pdb_id] = scaffold_key(
                args.refined_root / pdb_id / f"{pdb_id}_ligand.sdf"
            )
            protein_keys[pdb_id] = hashlib.sha1(
                sequences[pdb_id].encode()
            ).hexdigest()
            valid_ids.append(pdb_id)
        except Exception as error:
            failures.append({"pdb_id": pdb_id, "error": repr(error)})
    scaffold_split = grouped_split(valid_ids, scaffold_keys, args.seed)
    protein_split = grouped_split(valid_ids, protein_keys, args.seed)
    report = {
        "sample_count": len(valid_ids),
        "failures": failures,
        "scaffold": {
            "split_ids": scaffold_split,
            "group_count": len(set(scaffold_keys.values())),
            "overlap": verify(scaffold_split, scaffold_keys),
        },
        "exact_protein": {
            "split_ids": protein_split,
            "group_count": len(set(protein_keys.values())),
            "overlap": verify(protein_split, protein_keys),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "sample_count": report["sample_count"],
        "failure_count": len(failures),
        "scaffold_counts": {k: len(v) for k, v in scaffold_split.items()},
        "scaffold_overlap": report["scaffold"]["overlap"],
        "protein_counts": {k: len(v) for k, v in protein_split.items()},
        "protein_overlap": report["exact_protein"]["overlap"],
    }, indent=2))


if __name__ == "__main__":
    main()
