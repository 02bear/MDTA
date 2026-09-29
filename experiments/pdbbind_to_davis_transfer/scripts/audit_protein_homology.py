#!/usr/bin/env python3
"""Report Davis-to-clean-PDBbind protein homology; do not use it for drug-cold exclusion."""

import argparse
import csv
import json
import subprocess
from pathlib import Path


def read_fasta(path):
    result, name, pieces = {}, None, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                result[name] = "".join(pieces)
            name, pieces = line[1:].split()[0], []
        elif line:
            pieces.append(line)
    if name is not None:
        result[name] = "".join(pieces)
    return result


def write_fasta(records, path):
    with path.open("w") as handle:
        for name, sequence in records.items():
            handle.write(f">{name}\n{sequence}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, default=Path("/data1/ztx/MyModel-MDTA"))
    p.add_argument("--clean-split", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_all = read_fasta(args.project / "experiments/pdbbind_residue_transfer/data/homology30_cov80_sequences.fasta")
    clean_ids = set(json.loads(args.clean_split.read_text())["split_ids"]["train"])
    source = {key: value for key, value in source_all.items() if key in clean_ids}
    davis = {}
    with (args.project / "data/raw/davis/proteins.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            davis[str(row["protein_id"])] = row["sequence"]
    query = args.output_dir / "davis.fasta"
    target = args.output_dir / "pdbbind_fold1_clean_train.fasta"
    hits_path = args.output_dir / "hits.tsv"
    write_fasta(davis, query); write_fasta(source, target)
    mmseqs = args.project / "experiments/pdbbind_residue_transfer/tools/mmseqs/bin/mmseqs"
    tmp = args.output_dir / "mmseqs_tmp"
    command = [
        str(mmseqs), "easy-search", str(query), str(target), str(hits_path), str(tmp),
        "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0",
        "--max-seqs", "10000", "-s", "7.5",
        "--format-output", "query,target,fident,qcov,tcov,alnlen,qlen,tlen,evalue,bits",
    ]
    subprocess.run(command, check=True)
    kept = []
    for line in hits_path.read_text().splitlines():
        fields = line.split("\t")
        row = {
            "query": fields[0], "target": fields[1], "identity": float(fields[2]),
            "qcov": float(fields[3]), "tcov": float(fields[4]),
            "alignment_length": int(fields[5]), "query_length": int(fields[6]),
            "target_length": int(fields[7]), "evalue": float(fields[8]), "bits": float(fields[9]),
        }
        if row["identity"] >= 0.3 and row["qcov"] >= 0.8 and row["tcov"] >= 0.8:
            kept.append(row)
    report = {
        "davis_proteins": len(davis), "pdbbind_clean_train_proteins": len(source),
        "threshold": {"identity": 0.3, "query_coverage": 0.8, "target_coverage": 0.8},
        "qualifying_hits": len(kept),
        "davis_proteins_with_hit": len({x["query"] for x in kept}),
        "pdbbind_complexes_with_hit": len({x["target"] for x in kept}),
        "max_identity": max((x["identity"] for x in kept), default=None),
        "interpretation": "Reported only. Davis drug-cold train/validation/test reuse the same protein panel, so protein homology is not a heldout-axis leak.",
        "hits": kept,
    }
    (args.output_dir / "protein_homology_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "hits"}, indent=2))


if __name__ == "__main__":
    main()
