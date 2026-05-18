# -*- coding: utf-8 -*-
import time
import argparse
import requests
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Query UniProt accessions for base protein IDs")
    parser.add_argument("--input_csv", type=str, required=True, help="Stage1 csv path")
    parser.add_argument("--output_csv", type=str, required=True, help="Stage2 output csv path")
    parser.add_argument("--base_col", type=str, default="base_id", help="Base protein ID column")
    parser.add_argument("--species", type=str, default="9606", help="NCBI organism_id, e.g. 9606 for human")
    parser.add_argument("--size", type=int, default=5, help="UniProt returned hits per query")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between requests")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.base_col not in df.columns:
        raise ValueError(f"Column '{args.base_col}' not found in {args.input_csv}")

    base_ids = sorted(df[args.base_col].dropna().unique())

    rows = []
    session = requests.Session()

    for i, base in enumerate(base_ids, 1):
        # 先按 accession 精确查（KIBA 的 protein_id/base_id 多数本来就是 UniProt accession）
        # 再回退到 gene_exact 查询，兼容像 DAVIS 这种基因名风格的 ID。
        query_acc = f"accession:{base}"
        query_gene = f"gene_exact:{base} AND organism_id:{args.species} AND reviewed:true"
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": query_acc,
            "format": "tsv",
            "fields": "accession,gene_primary,protein_name,length,organism_name",
            "size": args.size,
        }

        try:
            r = session.get(url, params=params, timeout=30)
            r.raise_for_status()
            text = r.text.strip().splitlines()

            # accession 未命中时，回退 gene_exact
            if len(text) <= 1:
                params["query"] = query_gene
                r2 = session.get(url, params=params, timeout=30)
                r2.raise_for_status()
                text = r2.text.strip().splitlines()

            if len(text) <= 1:
                rows.append({
                    args.base_col: base,
                    "status": "no_hit",
                    "accession": "",
                    "gene_primary": "",
                    "protein_name": "",
                    "length": "",
                    "organism_name": "",
                })
            else:
                header = text[0].split("\t")
                first = text[1].split("\t")
                rec = dict(zip(header, first))
                rows.append({
                    args.base_col: base,
                    "status": "hit",
                    "accession": rec.get("Entry", ""),
                    "gene_primary": rec.get("Gene Names (primary)", ""),
                    "protein_name": rec.get("Protein names", ""),
                    "length": rec.get("Length", ""),
                    "organism_name": rec.get("Organism", ""),
                })

        except Exception as e:
            rows.append({
                args.base_col: base,
                "status": f"error: {e}",
                "accession": "",
                "gene_primary": "",
                "protein_name": "",
                "length": "",
                "organism_name": "",
            })

        if i % 20 == 0:
            print(f"[{i}/{len(base_ids)}] done")

        time.sleep(args.sleep)

    map_df = pd.DataFrame(rows)
    out = df.merge(map_df, on=args.base_col, how="left")
    out.to_csv(args.output_csv, index=False)

    print("\nSaved:", args.output_csv)
    cols_to_show = [c for c in ["protein_id", args.base_col, "accession", "status"] if c in out.columns]
    print(out[cols_to_show].head(20).to_string(index=False))
    print("\nStatus counts:")
    print(out["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
