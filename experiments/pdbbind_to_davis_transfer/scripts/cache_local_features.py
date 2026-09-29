#!/usr/bin/env python3
"""Cache frozen PDBbind-derived top-k atom-residue interaction vectors."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch


def stable_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % (2**31)


def pool_topk(residues, atoms, bias, scale, topk, temperature):
    logits = residues @ atoms.transpose(0, 1) * scale + bias
    flat = logits.flatten()
    k = min(topk, flat.numel())
    values, indices = torch.topk(flat, k=k, sorted=False)
    atom_count = atoms.shape[0]
    residue_idx = torch.div(indices, atom_count, rounding_mode="floor")
    atom_idx = indices.remainder(atom_count)
    weights = torch.softmax(values / temperature, dim=0)
    vector = (weights[:, None] * (residues[residue_idx] * atoms[atom_idx])).sum(0)
    return vector, {
        "all_logit_std": float(logits.std()),
        "normalized_logmeanexp": float(
            temperature * torch.logsumexp(flat / temperature, dim=0)
            - temperature * torch.log(flat.new_tensor(float(flat.numel())))
        ),
        "top_logit_max": float(values.max()),
        "top_logit_mean": float(values.mean()),
        "top_logit_std": float(values.std()),
        "residues": int(residues.shape[0]),
        "atoms": int(atoms.shape[0]),
    }


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, default=Path("/data1/ztx/MyModel-MDTA"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ligand-dir", type=Path, required=True)
    p.add_argument("--protein-cache", type=Path, required=True)
    p.add_argument("--pocket-cache", type=Path)
    p.add_argument("--max-pockets", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--random-seed", type=int, default=2026)
    args = p.parse_args()

    source_dir = args.project / "experiments/pdbbind_residue_transfer"
    sys.path.insert(0, str(source_dir))
    from typed_pair_model import TypedPairwiseContactPredictor

    pairs = pd.read_csv(args.project / "data/raw/davis/pairs.csv")
    pairs["drug_id"] = pairs["drug_id"].astype(str)
    pairs["protein_id"] = pairs["protein_id"].astype(str)
    drug_ids = list(dict.fromkeys(pairs["drug_id"]))
    protein_ids = list(dict.fromkeys(pairs["protein_id"]))
    manifest = json.loads((args.protein_cache / "manifest.json").read_text())["manifest"]
    pocket_data = (
        torch.load(args.pocket_cache, map_location="cpu", weights_only=False)
        if args.pocket_cache else None
    )

    pretrained = TypedPairwiseContactPredictor(hidden_dim=128, temperature=0.5).to(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    pretrained.load_state_dict(state)
    pretrained.eval()
    torch.manual_seed(args.random_seed)
    random_model = TypedPairwiseContactPredictor(hidden_dim=128, temperature=0.5).to(args.device)
    random_model.eval()
    for model in (pretrained, random_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    atom_pretrained, atom_random, atom_shuffle = {}, {}, {}
    for drug_id in drug_ids:
        graph = torch.load(args.ligand_dir / f"{drug_id}.pt", map_location=args.device, weights_only=False)
        atom_pretrained[drug_id] = pretrained.atom_projection(
            pretrained.ligand_encoder.forward_nodes(graph)
        ).cpu()
        atom_random[drug_id] = random_model.atom_projection(
            random_model.ligand_encoder.forward_nodes(graph)
        ).cpu()
        shuffled = graph.clone()
        generator = torch.Generator(device="cpu").manual_seed(stable_seed(drug_id))
        order = torch.randperm(graph.x.shape[0], generator=generator).to(graph.x.device)
        shuffled.x = graph.x[order]
        atom_shuffle[drug_id] = pretrained.atom_projection(
            pretrained.ligand_encoder.forward_nodes(shuffled)
        ).cpu()

    residue_pretrained, residue_random, selected_residues = {}, {}, {}
    for protein_id in protein_ids:
        entry = manifest[protein_id]
        obj = torch.load(entry["cache"], map_location="cpu", weights_only=False)
        embedding = obj["per_tok"].to(args.device)
        residue_pretrained[protein_id] = pretrained.protein_projection(embedding.float()).cpu()
        residue_random[protein_id] = random_model.protein_projection(embedding.float()).cpu()
        if pocket_data is None:
            indices = torch.arange(len(embedding), dtype=torch.long)
        else:
            entry_pockets = sorted(
                pocket_data[protein_id].get("pockets", []),
                key=lambda value: -float(value["score"]),
            )[:args.max_pockets]
            pieces = [p["sequence_indices"].long() for p in entry_pockets]
            indices = torch.unique(torch.cat(pieces), sorted=True) if pieces else torch.empty(0, dtype=torch.long)
            indices = indices[(indices >= 0) & (indices < len(embedding))]
            if len(indices) == 0:
                raise RuntimeError(f"no valid pocket residues: {protein_id}")
        selected_residues[protein_id] = indices

    output_by_pair = {}
    scalar_by_pair = {}
    diagnostics = {"pretrained": [], "random": [], "atom_shuffle": []}
    for p_index, protein_id in enumerate(protein_ids, 1):
        pocket_indices = selected_residues[protein_id]
        pre_res = residue_pretrained[protein_id][pocket_indices].to(args.device)
        rnd_res = residue_random[protein_id][pocket_indices].to(args.device)
        for drug_id in drug_ids:
            vectors, stats = {}, {}
            vectors["pretrained"], stats["pretrained"] = pool_topk(
                pre_res, atom_pretrained[drug_id].to(args.device),
                pretrained.base_pair_bias, pretrained.scale, args.topk, args.temperature,
            )
            vectors["random"], stats["random"] = pool_topk(
                rnd_res, atom_random[drug_id].to(args.device),
                random_model.base_pair_bias, random_model.scale, args.topk, args.temperature,
            )
            vectors["atom_shuffle"], stats["atom_shuffle"] = pool_topk(
                pre_res, atom_shuffle[drug_id].to(args.device),
                pretrained.base_pair_bias, pretrained.scale, args.topk, args.temperature,
            )
            output_by_pair[(drug_id, protein_id)] = {k: v.cpu() for k, v in vectors.items()}
            scalar_by_pair[(drug_id, protein_id)] = {
                k: stats[k]["normalized_logmeanexp"] for k in stats
            }
            for condition in diagnostics:
                diagnostics[condition].append(stats[condition])
        if p_index % 25 == 0:
            print(f"proteins={p_index}/{len(protein_ids)}", flush=True)

    tensors = {}
    for condition in ("pretrained", "random", "atom_shuffle"):
        tensors[condition] = torch.stack([
            output_by_pair[(row.drug_id, row.protein_id)][condition]
            for row in pairs.itertuples(index=False)
        ])
    mismatch_drug = {drug: drug_ids[(i + 1) % len(drug_ids)] for i, drug in enumerate(drug_ids)}
    tensors["mismatch"] = torch.stack([
        output_by_pair[(mismatch_drug[row.drug_id], row.protein_id)]["pretrained"]
        for row in pairs.itertuples(index=False)
    ])
    score_tensors = {}
    for condition in ("pretrained", "random", "atom_shuffle"):
        score_tensors[condition] = torch.tensor([
            scalar_by_pair[(row.drug_id, row.protein_id)][condition]
            for row in pairs.itertuples(index=False)
        ], dtype=torch.float32)
    score_tensors["mismatch"] = torch.tensor([
        scalar_by_pair[(mismatch_drug[row.drug_id], row.protein_id)]["pretrained"]
        for row in pairs.itertuples(index=False)
    ], dtype=torch.float32)

    audit = {
        "rows": len(pairs), "unique_drugs": len(drug_ids), "unique_proteins": len(protein_ids),
        "checkpoint": str(args.checkpoint), "topk": args.topk, "temperature": args.temperature,
        "pocket_cache": str(args.pocket_cache) if args.pocket_cache else None,
        "max_pockets": args.max_pockets if args.pocket_cache else None,
        "candidate_residue_count": {
            "min": min(len(x) for x in selected_residues.values()),
            "mean": sum(len(x) for x in selected_residues.values()) / len(selected_residues),
            "max": max(len(x) for x in selected_residues.values()),
        },
        "conditions": {}, "mismatch_mapping": mismatch_drug,
    }
    for condition, tensor in tensors.items():
        info = {
            "shape": list(tensor.shape),
            "finite": int(torch.isfinite(tensor).sum()),
            "feature_std_mean": float(tensor.std(dim=0).mean()),
            "sample_norm_std": float(tensor.norm(dim=1).std()),
        }
        if condition in diagnostics:
            info["mean_pairmap_logit_std"] = sum(x["all_logit_std"] for x in diagnostics[condition]) / len(diagnostics[condition])
            info["mean_top_logit_std"] = sum(x["top_logit_std"] for x in diagnostics[condition]) / len(diagnostics[condition])
        audit["conditions"][condition] = info
    audit["pretrained_vs_shuffle_mean_l2"] = float(
        (tensors["pretrained"] - tensors["atom_shuffle"]).norm(dim=1).mean()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **tensors,
        **{f"score_{key}": value for key, value in score_tensors.items()},
        "drug_id": pairs["drug_id"].tolist(),
        "protein_id": pairs["protein_id"].tolist(),
        "checkpoint": str(args.checkpoint),
    }, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
