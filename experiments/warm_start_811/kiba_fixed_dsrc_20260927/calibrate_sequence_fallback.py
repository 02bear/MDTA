#!/usr/bin/env python3
"""Calibrate KLIFS-pocket-to-full-sequence alignment on structure-mapped targets."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
import torch
from Bio.Align import PairwiseAligner


def align_pocket(full_sequence, pocket_sequence, gap_open, gap_extend):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    # Gaps in the query skip non-pocket residues in the full kinase sequence.
    aligner.query_internal_open_gap_score = gap_open
    aligner.query_internal_extend_gap_score = gap_extend
    aligner.query_left_open_gap_score = 0.0
    aligner.query_left_extend_gap_score = 0.0
    aligner.query_right_open_gap_score = 0.0
    aligner.query_right_extend_gap_score = 0.0
    # Dropping a standardized KLIFS position should be very expensive.
    aligner.target_internal_open_gap_score = -20.0
    aligner.target_internal_extend_gap_score = -2.0
    aligner.target_left_open_gap_score = -20.0
    aligner.target_left_extend_gap_score = -2.0
    aligner.target_right_open_gap_score = -20.0
    aligner.target_right_extend_gap_score = -2.0
    alignment = aligner.align(full_sequence, pocket_sequence)[0]
    result = [-1] * len(pocket_sequence)
    for (d0, d1), (p0, p1) in zip(*alignment.aligned):
        for offset in range(min(d1 - d0, p1 - p0)):
            result[p0 + offset] = d0 + offset
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--klifs-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, cache, output = args.project_root.resolve(), args.klifs_cache.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    proteins = pd.read_csv(root / "data/raw/davis/proteins.csv", dtype=str).set_index("protein_id")
    audit = pd.read_csv(cache.parent / "mapping_audit.csv")
    calibration_ids = audit.loc[
        (audit["mapped_positions"] >= 80) & (audit["pocket_reference_identity"].fillna(0) >= 0.9), "protein_id"
    ].tolist()

    rows = []
    for gap_open, gap_extend in itertools.product([-1.0, -2.0, -3.0, -5.0, -8.0], [-0.02, -0.05, -0.1, -0.25, -0.5]):
        exact = near1 = comparable = mapped = 0
        protein_exact = []
        for protein_id in calibration_ids:
            item = torch.load(cache / f"{protein_id}.pt", map_location="cpu", weights_only=False)
            truth = item["sequence_indices"].tolist()
            predicted = align_pocket(
                proteins.loc[protein_id, "sequence"], item["reference_pocket_sequence"], gap_open, gap_extend
            )
            valid = [(a, b) for a, b in zip(truth, predicted) if a >= 0 and b >= 0]
            if not valid:
                continue
            local_exact = sum(a == b for a, b in valid)
            exact += local_exact
            near1 += sum(abs(a - b) <= 1 for a, b in valid)
            comparable += len(valid)
            mapped += sum(x >= 0 for x in predicted)
            protein_exact.append(local_exact / len(valid))
        rows.append(
            {
                "query_gap_open": gap_open,
                "query_gap_extend": gap_extend,
                "n_proteins": len(protein_exact),
                "position_exact_accuracy": exact / max(1, comparable),
                "position_within1_accuracy": near1 / max(1, comparable),
                "median_protein_exact_accuracy": float(pd.Series(protein_exact).median()),
                "mean_mapped_positions": mapped / max(1, len(protein_exact)),
            }
        )
    results = pd.DataFrame(rows).sort_values(
        ["position_exact_accuracy", "median_protein_exact_accuracy"], ascending=False
    )
    results.to_csv(output / "sequence_fallback_grid.csv", index=False)
    best = results.iloc[0].to_dict()
    (output / "sequence_fallback_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(results.head(10).to_string(index=False))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
