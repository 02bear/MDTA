#!/usr/bin/env python3
"""Build an auditable KLIFS-85 -> Davis sequence/AFDB coordinate mapping.

Only target identities and structures are used.  No affinity labels or test split
membership are read by this preprocessing step.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import torch
from Bio.Align import PairwiseAligner
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB import PDBParser


API_ROOT = "https://klifs.net/api"

# The legacy AFDB manifest omitted these Davis display names although they are
# standard human kinases.  Values below were verified against KLIFS /kinase_ID.
ALIAS_ACCESSIONS = {
    "AMPK-alpha1": "Q13131",
    "AMPK-alpha2": "P54646",
    "CDK4-cyclinD1": "P11802",
    "CDK4-cyclinD3": "P11802",
    "IKK-alpha": "O15111",
    "IKK-beta": "O14920",
    "IKK-epsilon": "Q14164",
    "MRCKA": "Q5VT25",
    "MRCKB": "Q9Y5S2",
    "p38-alpha": "Q16539",
    "p38-beta": "Q15759",
    "p38-delta": "O15264",
    "p38-gamma": "P53778",
    "PFTAIRE2": "Q96Q40",
    "PKAC-alpha": "P17612",
    "PKAC-beta": "P22694",
    "S6K1": "P23443",
}


def request(session: requests.Session, endpoint: str, params: dict, attempts: int = 5):
    error = None
    for attempt in range(attempts):
        try:
            response = session.get(f"{API_ROOT}/{endpoint}", params=params, timeout=60)
            response.raise_for_status()
            return response
        except Exception as exc:  # transient KLIFS/network failures are retried
            error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"KLIFS request failed: {endpoint} {params}: {error}")


def batches(values, size):
    values = list(values)
    for start in range(0, len(values), size):
        yield values[start : start + size]


def numeric(value, default):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def choose_structure(rows):
    """Prefer complete, well processed, high-resolution pocket structures."""
    human = [row for row in rows if str(row.get("species", "")).lower() == "human"]
    candidates = human or rows
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            numeric(row.get("missing_residues"), 999),
            numeric(row.get("missing_atoms"), 99999),
            -numeric(row.get("quality_score"), -999),
            numeric(row.get("resolution"), 999),
            int(row["structure_ID"]),
        ),
    )


def residue_key(residue):
    _, number, insertion = residue.id
    insertion = str(insertion).strip()
    return f"{number}{insertion}" if insertion else str(number)


def chain_observed_residues(pdb_path: Path, chain_id: str):
    structure = PDBParser(QUIET=True).get_structure("klifs", str(pdb_path))
    model = next(structure.get_models())
    if chain_id not in model:
        raise KeyError(f"chain {chain_id!r} absent; available={list(model.child_dict)}")
    records = []
    for residue in model[chain_id]:
        if residue.id[0] != " " or "CA" not in residue:
            continue
        aa = protein_letters_3to1_extended.get(residue.resname.upper(), "X")
        records.append((residue_key(residue), aa))
    return records


def align_observed_to_davis(observed_sequence: str, davis_sequence: str):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -6.0
    aligner.extend_gap_score = -0.5
    # A crystallized kinase construct is usually an internal fragment of UniProt.
    aligner.query_left_open_gap_score = 0.0
    aligner.query_left_extend_gap_score = 0.0
    aligner.query_right_open_gap_score = 0.0
    aligner.query_right_extend_gap_score = 0.0
    alignment = aligner.align(davis_sequence, observed_sequence)[0]
    obs_to_davis = {}
    matches = 0
    aligned = 0
    for (d0, d1), (o0, o1) in zip(*alignment.aligned):
        block = min(d1 - d0, o1 - o0)
        for offset in range(block):
            di, oi = d0 + offset, o0 + offset
            obs_to_davis[oi] = di
            aligned += 1
            matches += int(davis_sequence[di] == observed_sequence[oi])
    return obs_to_davis, {
        "alignment_score": float(alignment.score),
        "aligned_residues": aligned,
        "observed_residues": len(observed_sequence),
        "observed_coverage": aligned / max(1, len(observed_sequence)),
        "aligned_identity": matches / max(1, aligned),
    }


def fetch_structure_assets(structure, raw_dir: Path):
    structure_id = int(structure["structure_ID"])
    mapping_path = raw_dir / "residue_maps" / f"{structure_id}.json"
    pdb_path = raw_dir / "pdb" / f"{structure_id}.pdb"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "KLIFS-Interact-P13D/1.0 (academic preprocessing)"
    if not mapping_path.exists():
        rows = request(session, "interactions_match_residues", {"structure_ID": structure_id}).json()
        mapping_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not pdb_path.exists():
        text = request(session, "structure_get_pdb_complex", {"structure_ID": structure_id}).text
        if not text.startswith(("HEADER", "TITLE", "REMARK", "ATOM")):
            raise RuntimeError(f"Unexpected PDB response for KLIFS structure {structure_id}")
        pdb_path.write_text(text, encoding="utf-8")
    return structure_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    raw_dir = output / "raw_klifs"
    cache_dir = output / "by_protein"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    proteins = pd.read_csv(root / "data/raw/davis/proteins.csv", dtype=str)
    manifest = pd.read_csv(root / "data/raw/davis/afdb_download_manifest.updated.csv", dtype=str)
    table = proteins.merge(manifest[["protein_id", "accession"]], on="protein_id", how="left", validate="one_to_one")
    table["accession"] = table["accession"].fillna(table["protein_id"].map(ALIAS_ACCESSIONS))

    session = requests.Session()
    session.headers["User-Agent"] = "KLIFS-Interact-P13D/1.0 (academic preprocessing)"
    accessions = sorted(table["accession"].dropna().unique())
    kinase_rows = []
    for batch in batches(accessions, 60):
        kinase_rows.extend(request(session, "kinase_ID", {"kinase_name": ",".join(batch), "species": "HUMAN"}).json())
    kinase_by_accession = {row.get("uniprot"): row for row in kinase_rows if row.get("uniprot")}

    kinase_ids = sorted({int(row["kinase_ID"]) for row in kinase_by_accession.values()})
    structures = []
    for batch in batches(kinase_ids, 30):
        structures.extend(request(session, "structures_list", {"kinase_ID": ",".join(map(str, batch))}).json())
    structures_by_kinase = {}
    for row in structures:
        structures_by_kinase.setdefault(int(row["kinase_ID"]), []).append(row)
    selected_by_accession = {
        accession: choose_structure(structures_by_kinase.get(int(info["kinase_ID"]), []))
        for accession, info in kinase_by_accession.items()
    }

    selected_unique = {int(row["structure_ID"]): row for row in selected_by_accession.values() if row}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_structure_assets, row, raw_dir) for row in selected_unique.values()]
        for future in as_completed(futures):
            future.result()

    audit_rows = []
    for item in table.itertuples(index=False):
        protein_id, sequence, accession = item.protein_id, item.sequence, item.accession
        kinase = kinase_by_accession.get(accession)
        structure = selected_by_accession.get(accession)
        record = {
            "protein_id": protein_id,
            "accession": accession,
            "sequence_length": len(sequence),
            "status": "ok",
        }
        if kinase is None:
            record.update(status="no_klifs_kinase", mapped_positions=0)
            audit_rows.append(record)
            continue
        if structure is None:
            record.update(status="no_klifs_structure", kinase=kinase["name"], mapped_positions=0)
            audit_rows.append(record)
            continue

        structure_id = int(structure["structure_ID"])
        try:
            observed = chain_observed_residues(raw_dir / "pdb" / f"{structure_id}.pdb", structure["chain"])
            observed_keys = [key for key, _ in observed]
            observed_sequence = "".join(aa for _, aa in observed)
            obs_to_davis, alignment_audit = align_observed_to_davis(observed_sequence, sequence)
            key_to_observed = {key: index for index, key in enumerate(observed_keys)}
            residue_map = json.loads((raw_dir / "residue_maps" / f"{structure_id}.json").read_text())
            indices = torch.full((85,), -1, dtype=torch.long)
            position_names = [""] * 85
            for row in residue_map:
                klifs_index = int(row["index"]) - 1
                xray = str(row.get("Xray_position", "_")).strip()
                position_names[klifs_index] = str(row.get("KLIFS_position", ""))
                observed_index = key_to_observed.get(xray)
                if observed_index is not None and observed_index in obs_to_davis:
                    indices[klifs_index] = obs_to_davis[observed_index]

            gvp = torch.load(root / "data/processed/davis/protein_3d_gvp" / f"{protein_id}.pt", map_location="cpu", weights_only=False)
            coords = torch.zeros((85, 3), dtype=torch.float32)
            valid = (indices >= 0) & (indices < len(gvp["coords"]))
            coords[valid] = gvp["coords"][indices[valid]].float()
            actual_pocket = "".join(sequence[i] if i >= 0 else "-" for i in indices.tolist())
            reference_pocket = str(kinase.get("pocket") or "")
            reference_matches = sum(a == b for a, b, keep in zip(actual_pocket, reference_pocket, valid.tolist()) if keep)
            reference_identity = reference_matches / max(1, int(valid.sum()))

            payload = {
                "protein_id": protein_id,
                "accession": accession,
                "kinase_name": kinase["name"],
                "kinase_id": int(kinase["kinase_ID"]),
                "structure_id": structure_id,
                "pdb": structure["pdb"],
                "chain": structure["chain"],
                "sequence_indices": indices,
                "mask": valid,
                "coords": coords,
                "pocket_sequence": actual_pocket,
                "reference_pocket_sequence": reference_pocket,
                "klifs_position_names": position_names,
            }
            torch.save(payload, cache_dir / f"{protein_id}.pt")
            record.update(
                kinase=kinase["name"],
                kinase_id=int(kinase["kinase_ID"]),
                structure_id=structure_id,
                pdb=structure["pdb"],
                chain=structure["chain"],
                structure_missing_residues=int(numeric(structure.get("missing_residues"), 999)),
                structure_missing_atoms=int(numeric(structure.get("missing_atoms"), 99999)),
                mapped_positions=int(valid.sum()),
                pocket_coverage=float(valid.float().mean()),
                pocket_reference_identity=reference_identity,
                **alignment_audit,
            )
            if alignment_audit["aligned_identity"] < 0.85 or int(valid.sum()) < 70:
                record["status"] = "low_quality_mapping"
        except Exception as exc:
            record.update(status="mapping_error", mapped_positions=0, error=repr(exc))
        audit_rows.append(record)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output / "mapping_audit.csv", index=False)
    summary = {
        "n_davis_proteins": len(table),
        "n_unique_accessions": len(accessions),
        "n_klifs_accessions": len(kinase_by_accession),
        "n_selected_structures": len(selected_unique),
        "status_counts": audit["status"].value_counts(dropna=False).to_dict(),
        "mapped_positions": audit["mapped_positions"].describe().to_dict(),
        "n_ge_80_positions": int((audit["mapped_positions"] >= 80).sum()),
        "n_ge_70_positions": int((audit["mapped_positions"] >= 70).sum()),
    }
    (output / "mapping_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
