#!/usr/bin/env python3
"""Create full-length Davis ProtT5 residue embeddings compatible with PDBbind."""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import T5EncoderModel, T5Tokenizer


AA_REPLACE = str.maketrans({"U": "X", "Z": "X", "O": "X", "B": "X"})


def normalize(seq):
    seq = seq.strip().upper().translate(AA_REPLACE)
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX]", "X", seq)


def spans(length, chunk_size, overlap):
    if length <= chunk_size:
        return [(0, length)]
    step = chunk_size - overlap
    result, start = [], 0
    while start < length:
        end = min(start + chunk_size, length)
        result.append((start, end))
        if end == length:
            break
        start += step
    return result


def chunk_weights(size, overlap, left_overlap, right_overlap, device):
    weight = torch.ones(size, device=device, dtype=torch.float32)
    if left_overlap:
        n = min(overlap, size)
        weight[:n] = torch.linspace(1.0 / (n + 1), 1.0, n, device=device)
    if right_overlap:
        n = min(overlap, size)
        weight[-n:] = torch.minimum(
            weight[-n:], torch.linspace(1.0, 1.0 / (n + 1), n, device=device)
        )
    return weight


@torch.no_grad()
def embed_sequence(seq, tokenizer, model, device, chunk_size, overlap):
    accumulator = torch.zeros((len(seq), model.config.d_model), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((len(seq), 1), dtype=torch.float32, device=device)
    sequence_spans = spans(len(seq), chunk_size, overlap)
    for index, (start, end) in enumerate(sequence_spans):
        part = seq[start:end]
        encoded = tokenizer(
            " ".join(part), return_tensors="pt", add_special_tokens=True,
            padding=False, truncation=False
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        hidden = model(**encoded).last_hidden_state.squeeze(0)
        mask = encoded["attention_mask"].squeeze(0).bool()
        hidden = hidden[mask]
        if hidden.shape[0] == len(part) + 1:
            hidden = hidden[:-1]
        elif hidden.shape[0] == len(part) + 2:
            hidden = hidden[1:-1]
        if hidden.shape[0] != len(part):
            raise RuntimeError(f"token mismatch at {start}:{end}: {hidden.shape[0]} != {len(part)}")
        hidden = hidden.float()
        weight = chunk_weights(
            len(part), overlap, index > 0, index + 1 < len(sequence_spans), device
        )[:, None]
        accumulator[start:end] += hidden * weight
        weight_sum[start:end] += weight
    if torch.any(weight_sum == 0):
        raise RuntimeError("one or more residues received no embedding")
    return (accumulator / weight_sum).cpu(), sequence_spans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--proteins-csv", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--chunk-size", type=int, default=900)
    p.add_argument("--overlap", type=int, default=128)
    args = p.parse_args()
    if args.overlap >= args.chunk_size:
        raise ValueError("overlap must be smaller than chunk-size")

    by_sequence = args.output_dir / "by_sequence"
    by_sequence.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.proteins_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append((str(row["protein_id"]), normalize(row["sequence"])))
    grouped = defaultdict(list)
    for protein_id, seq in rows:
        grouped[hashlib.sha256(seq.encode()).hexdigest()].append((protein_id, seq))

    tokenizer = T5Tokenizer.from_pretrained(str(args.model), do_lower_case=False, local_files_only=True)
    model = T5EncoderModel.from_pretrained(str(args.model), local_files_only=True)
    model.eval().to(args.device)
    if str(args.device).startswith("cuda"):
        model.half()

    manifest, failures = {}, []
    for number, (digest, entries) in enumerate(grouped.items(), 1):
        seq = entries[0][1]
        output = by_sequence / f"{digest}.pt"
        try:
            if output.exists():
                obj = torch.load(output, map_location="cpu", weights_only=False)
                embedding = obj["per_tok"]
                sequence_spans = obj["spans"]
                if embedding.shape != (len(seq), model.config.d_model):
                    raise RuntimeError(f"bad existing cache shape {tuple(embedding.shape)}")
            else:
                embedding, sequence_spans = embed_sequence(
                    seq, tokenizer, model, torch.device(args.device), args.chunk_size, args.overlap
                )
                torch.save({
                    "sequence_hash": digest,
                    "sequence": seq,
                    "per_tok": embedding,
                    "spans": sequence_spans,
                    "model": str(args.model),
                }, output)
            for protein_id, _ in entries:
                manifest[protein_id] = {
                    "sequence_hash": digest,
                    "sequence_length": len(seq),
                    "embedding_shape": list(embedding.shape),
                    "chunks": len(sequence_spans),
                    "cache": str(output),
                }
            print(f"[{number}/{len(grouped)}] {digest[:10]} length={len(seq)} ids={len(entries)} chunks={len(sequence_spans)}", flush=True)
        except Exception as exc:
            for protein_id, _ in entries:
                failures.append({"protein_id": protein_id, "sequence_hash": digest, "error": str(exc)})

    audit = {
        "purpose": "PDBbind-compatible 1024d per-residue ProtT5 inputs without truncating Davis proteins.",
        "input": str(args.proteins_csv),
        "output": str(args.output_dir),
        "model": str(args.model),
        "normalization": "uppercase; U/Z/O/B and non-standard residues -> X; one token per residue",
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "overlap_merge": "linear edge weights followed by weighted mean",
        "protein_ids": len(rows),
        "unique_sequences": len(grouped),
        "success_ids": len(manifest),
        "failures": failures,
        "all_lengths_exact": all(v["embedding_shape"] == [v["sequence_length"], 1024] for v in manifest.values()),
        "max_sequence_length": max(len(seq) for _, seq in rows),
        "manifest": manifest,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k != "manifest"}, indent=2), flush=True)
    if failures or len(manifest) != len(rows) or not audit["all_lengths_exact"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
