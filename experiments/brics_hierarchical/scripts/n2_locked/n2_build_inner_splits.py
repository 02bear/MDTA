#!/usr/bin/env python3
"""Materialize the preregistered, label-independent N2 inner drug splits."""

import argparse
import hashlib
import json
import math
from pathlib import Path


SALT = "N2_INNER_V1"


def ranked(drug_id: str, fold: int) -> str:
    value = f"{SALT}|fold={fold}|drug={drug_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    for fold in range(2, 6):
        source = args.split_root / f"fold_{fold}" / "split.json"
        outer = json.loads(source.read_text(encoding="utf-8"))
        train_drugs = [str(x) for x in outer["train_drugs"]]
        ordered = sorted(train_drugs, key=lambda x: ranked(x, fold))
        n_val = math.ceil(len(ordered) / 5)
        inner_val_drugs = ordered[:n_val]
        inner_train_drugs = ordered[n_val:]
        row_drugs = [str(x) for x in outer.get("all_drug_ids", [])]
        if row_drugs:
            raise RuntimeError("Unexpected all_drug_ids field; row mapping must come from pair IDs")

        # Davis split files already carry complete pair indices by drug.  Derive the
        # inner row lists from the canonical pairs file later, after checking IDs.
        payload = {
            "protocol": "N2_INNER_V1_SHA256_DRUG_ID",
            "fold": fold,
            "source_outer_split": str(source),
            "label_used": False,
            "inner_train_drugs": inner_train_drugs,
            "inner_val_drugs": inner_val_drugs,
            "counts": {"inner_train_drugs": len(inner_train_drugs), "inner_val_drugs": len(inner_val_drugs)},
        }
        train, val, outer_train = set(inner_train_drugs), set(inner_val_drugs), set(train_drugs)
        assert not train & val
        assert train | val == outer_train
        output = args.output_root / f"fold_{fold}" / "inner_split.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output), **payload["counts"]}), flush=True)


if __name__ == "__main__":
    main()
