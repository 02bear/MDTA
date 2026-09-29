#!/usr/bin/env python3
"""Materialize the 25 locked N3 cross-fit splits without reading labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from n3_common import build_n3_split, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for fold in range(1, 6):
        seen = []
        for cf in range(1, 6):
            split = build_n3_split(args.project, fold, cf)
            seen.extend(split["holdout_drugs"])
            path = args.output / f"fold_{fold}" / f"cf_{cf}.json"
            write_json(path, split)
            records.append({"fold": fold, "cf": cf, "path": str(path)})
        outer = build_n3_split(args.project, fold, 1)["outer_train_drugs"]
        assert len(seen) == len(set(seen)) == len(outer)
        assert set(seen) == set(outer)
    manifest = {
        "label_used": False,
        "run_count": len(records),
        "all_outer_train_drugs_held_out_exactly_once": True,
        "records": records,
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
