# -*- coding: utf-8 -*-
import time
from pathlib import Path
import requests
import pandas as pd

INPUT = "data/raw/davis/protein_mapping_stage1.csv"
OUTPUT = "data/raw/davis/protein_mapping_stage2_uniprot.csv"

df = pd.read_csv(INPUT)

base_ids = sorted(df["base_id"].dropna().unique())

rows = []
session = requests.Session()

for i, base in enumerate(base_ids, 1):
    query = f'gene_exact:{base} AND organism_id:9606 AND reviewed:true'
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,gene_primary,protein_name,length,organism_name",
        "size": 5,
    }

    try:
        r = session.get(url, params=params, timeout=30)
        r.raise_for_status()
        text = r.text.strip().splitlines()

        if len(text) <= 1:
            rows.append({
                "base_id": base,
                "status": "no_hit",
                "accession": "",
                "gene_primary": "",
                "protein_name": "",
                "length": "",
                "organism_name": "",
            })
        else:
            # 第一行是 header
            header = text[0].split("\t")
            first = text[1].split("\t")
            rec = dict(zip(header, first))
            rows.append({
                "base_id": base,
                "status": "hit",
                "accession": rec.get("Entry", ""),
                "gene_primary": rec.get("Gene Names (primary)", ""),
                "protein_name": rec.get("Protein names", ""),
                "length": rec.get("Length", ""),
                "organism_name": rec.get("Organism", ""),
            })

    except Exception as e:
        rows.append({
            "base_id": base,
            "status": f"error: {e}",
            "accession": "",
            "gene_primary": "",
            "protein_name": "",
            "length": "",
            "organism_name": "",
        })

    if i % 20 == 0:
        print(f"[{i}/{len(base_ids)}] done")

    time.sleep(0.2)

map_df = pd.DataFrame(rows)
out = df.merge(map_df, on="base_id", how="left")
out.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
print(out[["protein_id", "base_id", "accession", "status"]].head(20).to_string(index=False))
print("\nStatus counts:")
print(out["status"].value_counts(dropna=False))