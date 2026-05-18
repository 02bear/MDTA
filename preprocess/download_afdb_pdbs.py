# -*- coding: utf-8 -*-
import os
import time
import requests
import pandas as pd

INPUT = "data/raw/davis/protein_mapping_stage2_uniprot.csv"
OUTDIR = "data/raw/davis/pdb_afdb"
MANIFEST = "data/raw/davis/afdb_download_manifest.csv"

os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(INPUT)

# 只按 unique protein_id 下
df = df[["protein_id", "base_id", "accession", "status"]].drop_duplicates()

rows = []
session = requests.Session()

for i, row in enumerate(df.itertuples(index=False), 1):
    protein_id = row.protein_id
    accession = str(row.accession) if pd.notna(row.accession) else ""

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

        # 取第一个模型
        rec = meta[0]

        # 不同版本字段名可能略有差别，优先找 pdbUrl
        pdb_url = rec.get("pdbUrl", "")
        if not pdb_url:
            rows.append({
                "protein_id": protein_id,
                "accession": accession,
                "download_status": "no_pdb_url",
                "pdb_path": "",
            })
            continue

        save_path = str(OUTDIR / f"{protein_id}.pdb")
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

    time.sleep(0.2)

manifest = pd.DataFrame(rows)
manifest.to_csv(MANIFEST, index=False)

print("\nSaved manifest:", MANIFEST)
print(manifest["download_status"].value_counts(dropna=False))