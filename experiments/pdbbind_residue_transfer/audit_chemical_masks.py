#!/usr/bin/env python3
"""Measure proposed chemistry-mask label coverage before using any hard mask."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from chemical_masks import (
    INTERACTION_TYPES,
    atom_candidate_mask,
    pair_candidate_mask,
    parse_site_sequences,
    residue_candidate_mask,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtbind-root", type=Path, required=True)
    parser.add_argument("--pair-labels", type=Path, required=True)
    parser.add_argument("--ligand-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    site = args.dtbind_root / "Data" / "site"
    sequences = parse_site_sequences(site / "site_labels.txt")
    payload = json.loads(args.pair_labels.read_text())
    stats = defaultdict(lambda: {
        "positive_total": 0,
        "positive_atom_allowed": 0,
        "positive_residue_allowed": 0,
        "positive_pair_allowed": 0,
        "candidate_pairs": 0,
        "all_pairs": 0,
        "complexes_with_positive": 0,
    })
    skipped = []

    for record in payload["records"]:
        pdb_id = record["pdb_id"]
        sequence = sequences.get(pdb_id)
        ligand_path = (
            args.ligand_cache / f"{pdb_id}_ligand_rich.pt"
            if args.ligand_cache
            else site / "ligand_graph" / f"{pdb_id}_ligand.pt"
        )
        if sequence is None or not ligand_path.exists() or len(sequence) != record["residue_count"]:
            skipped.append(pdb_id)
            continue
        ligand = torch.load(ligand_path, map_location="cpu", weights_only=False)
        if ligand.x.shape[0] != record["atom_count"]:
            skipped.append(pdb_id)
            continue
        pairs_by_type = defaultdict(set)
        for pair in record["pairs"]:
            pairs_by_type[str(pair["type"])].add((int(pair["residue_index"]), int(pair["atom_index"])))
        for interaction_type in INTERACTION_TYPES:
            mask = pair_candidate_mask(ligand.x, sequence, interaction_type)
            atom_mask = atom_candidate_mask(ligand.x, interaction_type)
            residue_mask = residue_candidate_mask(sequence, interaction_type)
            current = stats[interaction_type]
            current["candidate_pairs"] += int(mask.sum())
            current["all_pairs"] += int(mask.numel())
            positives = pairs_by_type.get(interaction_type, set())
            if positives:
                current["complexes_with_positive"] += 1
            for residue, atom in positives:
                current["positive_total"] += 1
                current["positive_atom_allowed"] += int(atom_mask[atom])
                current["positive_residue_allowed"] += int(residue_mask[residue])
                current["positive_pair_allowed"] += int(mask[residue, atom])

    report = {"skipped_count": len(skipped), "skipped": skipped, "types": {}}
    for interaction_type in INTERACTION_TYPES:
        current = dict(stats[interaction_type])
        total = current["positive_total"]
        current["atom_positive_coverage"] = current["positive_atom_allowed"] / total if total else None
        current["residue_positive_coverage"] = current["positive_residue_allowed"] / total if total else None
        current["pair_positive_coverage"] = current["positive_pair_allowed"] / total if total else None
        current["candidate_pair_fraction"] = current["candidate_pairs"] / current["all_pairs"] if current["all_pairs"] else None
        report["types"][interaction_type] = current
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
