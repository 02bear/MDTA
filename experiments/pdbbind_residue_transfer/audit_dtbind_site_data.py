#!/usr/bin/env python3
"""Audit existing DTBind PDBbind/PLIP residue-site labels without modifying them."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_label_file(path: Path):
    lines = [line.strip() for line in path.open(encoding="utf-8") if line.strip()]
    if len(lines) % 3:
        raise ValueError(f"Expected 3 non-empty lines per record, got {len(lines)}")
    records = {}
    duplicate_ids = []
    for offset in range(0, len(lines), 3):
        header, sequence, labels = lines[offset : offset + 3]
        if not header.startswith(">"):
            raise ValueError(f"Malformed header at record {offset // 3}: {header[:40]}")
        pdb_id = header[1:].lower()
        if pdb_id in records:
            duplicate_ids.append(pdb_id)
        records[pdb_id] = {"sequence": sequence, "labels": labels}
    return records, duplicate_ids


def read_ids(path: Path):
    if not path.exists():
        return []
    return [line.strip().lower() for line in path.open() if line.strip()]


def summarize(args):
    site_dir = args.dtbind_root / "Data" / "site"
    records, duplicate_ids = parse_label_file(site_dir / "site_labels.txt")
    splits = {
        name: read_ids(site_dir / f"{name}_ids.txt")
        for name in ("train", "val", "test")
    }
    split_sets = {name: set(ids) for name, ids in splits.items()}

    protein_graph_ids = {
        path.stem.lower() for path in (site_dir / "protein_graph").glob("*.pt")
    }
    ligand_graph_ids = {
        path.name.lower().removesuffix("_ligand.pt")
        for path in (site_dir / "ligand_graph").glob("*_ligand.pt")
    }
    refined_ids = {
        path.parent.name.lower()
        for path in args.pdbbind_refined.glob("*/*_protein.pdb")
    }

    valid_lengths = []
    invalid_lengths = []
    zero_positive = []
    positive_counts = []
    positive_rates = []
    invalid_characters = []
    sequence_counts = Counter()
    rows = []
    for pdb_id, record in records.items():
        seq = record["sequence"]
        labels = record["labels"]
        sequence_counts[seq] += 1
        label_chars = set(labels)
        if not label_chars <= {"0", "1"}:
            invalid_characters.append({"pdb_id": pdb_id, "chars": sorted(label_chars)})
        length_ok = len(seq) == len(labels)
        positives = labels.count("1")
        if length_ok:
            valid_lengths.append(pdb_id)
        else:
            invalid_lengths.append(
                {"pdb_id": pdb_id, "sequence_len": len(seq), "label_len": len(labels)}
            )
        if positives == 0:
            zero_positive.append(pdb_id)
        positive_counts.append(positives)
        positive_rates.append(positives / max(len(labels), 1))
        rows.append(
            {
                "pdb_id": pdb_id,
                "sequence_len": len(seq),
                "label_len": len(labels),
                "positive_count": positives,
                "positive_rate": positives / max(len(labels), 1),
                "has_protein_graph": pdb_id in protein_graph_ids,
                "has_ligand_graph": pdb_id in ligand_graph_ids,
                "in_refined_set": pdb_id in refined_ids,
            }
        )

    all_split_ids = set().union(*split_sets.values())
    split_overlaps = {
        "train_val": sorted(split_sets["train"] & split_sets["val"]),
        "train_test": sorted(split_sets["train"] & split_sets["test"]),
        "val_test": sorted(split_sets["val"] & split_sets["test"]),
    }
    duplicate_sequence_groups = sum(count > 1 for count in sequence_counts.values())

    def stats(values):
        values = sorted(values)
        if not values:
            return {}
        n = len(values)
        return {
            "min": values[0],
            "median": values[n // 2],
            "mean": sum(values) / n,
            "max": values[-1],
        }

    report = {
        "record_count": len(records),
        "duplicate_record_ids": duplicate_ids,
        "valid_length_count": len(valid_lengths),
        "invalid_lengths": invalid_lengths[:100],
        "invalid_label_characters": invalid_characters[:100],
        "zero_positive_count": len(zero_positive),
        "zero_positive_examples": zero_positive[:30],
        "positive_count_stats": stats(positive_counts),
        "positive_rate_stats": stats(positive_rates),
        "unique_sequence_count": len(sequence_counts),
        "duplicate_sequence_group_count": duplicate_sequence_groups,
        "protein_graph_count": len(protein_graph_ids),
        "ligand_graph_count": len(ligand_graph_ids),
        "both_graphs_and_labels_count": len(set(records) & protein_graph_ids & ligand_graph_ids),
        "refined_set_count": len(refined_ids),
        "refined_with_labels_count": len(refined_ids & set(records)),
        "refined_with_labels_and_graphs_count": len(
            refined_ids & set(records) & protein_graph_ids & ligand_graph_ids
        ),
        "splits": {
            name: {
                "raw_count": len(ids),
                "unique_count": len(ids_set),
                "with_labels": len(ids_set & set(records)),
                "with_both_graphs": len(ids_set & protein_graph_ids & ligand_graph_ids),
            }
            for (name, ids), ids_set in zip(splits.items(), split_sets.values())
        },
        "split_union_count": len(all_split_ids),
        "split_union_with_labels_count": len(all_split_ids & set(records)),
        "split_overlaps": split_overlaps,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    columns = list(rows[0]) if rows else []
    with (args.output_dir / "audit_manifest.csv").open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(str(row[column]) for column in columns) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtbind-root", type=Path, required=True)
    parser.add_argument("--pdbbind-refined", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    summarize(parser.parse_args())


if __name__ == "__main__":
    main()
