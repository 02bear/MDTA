#!/usr/bin/env python3
"""Merge independently generated PLIP label shards with integrity checks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="part_*.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.shard_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No shards matched {args.shard_dir / args.pattern}")
    records, failures = [], []
    for path in files:
        payload = json.loads(path.read_text())
        records.extend(payload["records"])
        failures.extend(payload["failures"])
    record_counts = Counter(record["pdb_id"] for record in records)
    failure_counts = Counter(record["pdb_id"] for record in failures)
    duplicate_records = sorted(key for key, value in record_counts.items() if value != 1)
    duplicate_failures = sorted(key for key, value in failure_counts.items() if value != 1)
    overlap = sorted(set(record_counts) & set(failure_counts))
    if duplicate_records or duplicate_failures or overlap:
        raise ValueError(
            {"duplicate_records": duplicate_records, "duplicate_failures": duplicate_failures, "overlap": overlap}
        )
    interaction_counts = Counter(
        pair["type"] for record in records for pair in record["pairs"]
    )
    report = {
        "records": sorted(records, key=lambda item: item["pdb_id"]),
        "failures": sorted(failures, key=lambda item: item["pdb_id"]),
        "summary": {
            "shard_count": len(files),
            "success_count": len(records),
            "failure_count": len(failures),
            "total_count": len(records) + len(failures),
            "interaction_counts": dict(sorted(interaction_counts.items())),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
