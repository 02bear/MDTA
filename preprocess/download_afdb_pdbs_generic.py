# -*- coding: utf-8 -*-
import os
import time
import argparse
import requests
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Download AFDB PDBs from UniProt accession list")
    parser.add_argument("--input_csv", type=str, required=True, help="Stage2 UniProt mapping CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save AFDB PDB files")
    parser.add_argument("--manifest_csv", type=str, required=True, help="Path to save download manifest CSV")
    parser.add_argument("--protein_col", type=str, default="protein_id", help="Protein ID column")
    parser.add_argument("--accession_col", type=str, default="accession", help="UniProt accession column")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between requests")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    required_cols = [args.protein_col, args.accession_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {args.input_csv}: {missing}")

    df = df[required_cols].drop_duplicates()

    rows = []
    session = requests.Session()

    for i, row in enumerate(df.itertuples(index=False), 1):
        protein_id = str(getattr(row, args.protein_col))
        accession = str(getattr(row, args.accession_col)) if pd.notna(getattr(row, args.accession_col)) else ""

        if not accession or accession.strip() == "":
            rows.append({
                "protein_id": protein_id,
                "accession": accession,
                "download_status": "skip_no_accession",
                "pdb_path": "",
            })
            continue

        try:
            meta_url = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
            r = session.get(meta_url, timeout=30)
            r.raise_for_status()
            meta = r.json()

            if not meta:
                rows.append({
                    "protein_id": protein_id,
                    "accession": accession,
                    "download_status": "no_afdb_entry",
                    "pdb_path": "",
                })
                continue

            rec = meta[0]
            pdb_url = rec.get("pdbUrl", "")
            if not pdb_url:
                rows.append({
                    "protein_id": protein_id,
                    "accession": accession,
                    "download_status": "no_pdb_url",
                    "pdb_path": "",
                })
                continue

            save_path = os.path.join(args.output_dir, f"{protein_id}.pdb")
            rr = session.get(pdb_url, timeout=60)
            rr.raise_for_status()

            with open(save_path, "wb") as f:
                f.write(rr.content)

            rows.append({
                "protein_id": protein_id,
                "accession": accession,
                "download_status": "ok",
                "pdb_path": save_path,
            })

        except Exception as e:
            rows.append({
                "protein_id": protein_id,
                "accession": accession,
                "download_status": f"error: {e}",
                "pdb_path": "",
            })

        if i % 20 == 0:
            print(f"[{i}/{len(df)}] done")

        time.sleep(args.sleep)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.manifest_csv, index=False)

    print("\nSaved manifest:", args.manifest_csv)
    print(manifest["download_status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
