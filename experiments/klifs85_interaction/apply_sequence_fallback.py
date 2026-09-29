#!/usr/bin/env python3
"""Apply the structure-calibrated sequence fallback to uncovered KLIFS targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests
import torch

from calibrate_sequence_fallback import align_pocket
from prepare_klifs85 import ALIAS_ACCESSIONS, batches, request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--klifs-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.project_root.resolve(), args.klifs_dir.resolve()
    cache_dir = output / "by_protein"

    proteins = pd.read_csv(root / "data/raw/davis/proteins.csv", dtype=str)
    manifest = pd.read_csv(root / "data/raw/davis/afdb_download_manifest.updated.csv", dtype=str)
    table = proteins.merge(manifest[["protein_id", "accession"]], on="protein_id", how="left")
    table["accession"] = table["accession"].fillna(table["protein_id"].map(ALIAS_ACCESSIONS))
    table = table.set_index("protein_id")
    audit = pd.read_csv(output / "mapping_audit.csv").set_index("protein_id")
    fallback_ids = audit.index[audit["status"] != "ok"].tolist()
    accessions = sorted(table.loc[fallback_ids, "accession"].dropna().unique())

    session = requests.Session()
    session.headers["User-Agent"] = "KLIFS-Interact-P13D/1.0 (academic preprocessing)"
    rows = []
    for batch in batches(accessions, 60):
        rows.extend(request(session, "kinase_ID", {"kinase_name": ",".join(batch), "species": "HUMAN"}).json())
    kinase_by_accession = {row.get("uniprot"): row for row in rows if row.get("uniprot")}

    position_names = [""] * 85
    residue_maps = sorted((output / "raw_klifs/residue_maps").glob("*.json"))
    if residue_maps:
        for row in json.loads(residue_maps[0].read_text()):
            position_names[int(row["index"]) - 1] = str(row.get("KLIFS_position", ""))

    for protein_id in fallback_ids:
        accession = table.loc[protein_id, "accession"]
        kinase = kinase_by_accession.get(accession)
        if kinase is None or len(str(kinase.get("pocket") or "")) != 85:
            continue
        sequence = table.loc[protein_id, "sequence"]
        reference = str(kinase["pocket"])
        predicted = align_pocket(sequence, reference, gap_open=-5.0, gap_extend=-0.02)
        indices = torch.tensor(predicted, dtype=torch.long)
        gvp = torch.load(
            root / "data/processed/davis/protein_3d_gvp" / f"{protein_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        valid = (indices >= 0) & (indices < len(gvp["coords"]))
        coords = torch.zeros((85, 3), dtype=torch.float32)
        coords[valid] = gvp["coords"][indices[valid]].float()
        actual = "".join(sequence[i] if i >= 0 else "-" for i in indices.tolist())
        identity = sum(a == b for a, b, keep in zip(actual, reference, valid.tolist()) if keep) / max(1, int(valid.sum()))
        old = torch.load(cache_dir / f"{protein_id}.pt", map_location="cpu", weights_only=False) if (cache_dir / f"{protein_id}.pt").exists() else None
        payload = {
            "protein_id": protein_id,
            "accession": accession,
            "kinase_name": kinase["name"],
            "kinase_id": int(kinase["kinase_ID"]),
            "structure_id": None,
            "pdb": None,
            "chain": None,
            "mapping_method": "sequence_fallback_calibrated",
            "fallback_calibration_position_exact_accuracy": 0.9916703375916063,
            "sequence_indices": indices,
            "mask": valid,
            "coords": coords,
            "pocket_sequence": actual,
            "reference_pocket_sequence": reference,
            "klifs_position_names": position_names,
        }
        if old is not None:
            payload["replaced_mapping"] = {
                key: old.get(key) for key in ("structure_id", "pdb", "chain")
            }
        torch.save(payload, cache_dir / f"{protein_id}.pt")
        status = "ok_sequence_fallback" if int(valid.sum()) >= 80 and identity >= 0.75 else "low_quality_sequence_fallback"
        audit.loc[protein_id, "status"] = status
        audit.loc[protein_id, "kinase"] = kinase["name"]
        audit.loc[protein_id, "kinase_id"] = int(kinase["kinase_ID"])
        audit.loc[protein_id, "mapping_method"] = "sequence_fallback_calibrated"
        audit.loc[protein_id, "mapped_positions"] = int(valid.sum())
        audit.loc[protein_id, "pocket_coverage"] = float(valid.float().mean())
        audit.loc[protein_id, "pocket_reference_identity"] = identity

    audit = audit.reset_index()
    audit.to_csv(output / "mapping_audit.csv", index=False)
    summary = {
        "n_davis_proteins": len(audit),
        "status_counts": audit["status"].value_counts(dropna=False).to_dict(),
        "mapped_positions": audit["mapped_positions"].describe().to_dict(),
        "n_ge_80_positions": int((audit["mapped_positions"] >= 80).sum()),
        "n_ge_70_positions": int((audit["mapped_positions"] >= 70).sum()),
        "sequence_fallback_calibration": {
            "n_structure_mapped_proteins": 295,
            "position_exact_accuracy": 0.9916703375916063,
            "position_within1_accuracy": 0.993071963477634,
            "query_gap_open": -5.0,
            "query_gap_extend": -0.02,
        },
    }
    (output / "mapping_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
