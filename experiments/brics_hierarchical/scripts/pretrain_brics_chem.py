#!/usr/bin/env python3
"""Pretrain BRICS chemistry on unique training drugs before affinity fitting."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_brics_hierarchical_stage1 as base
from train_brics_chem_stage1 import BRICSChemHierarchicalP13D, ChemStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/data1/ztx/MyModel-MDTA"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--global-cache", type=Path, required=True)
    parser.add_argument("--drug-cache", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--condition", choices=["real", "random_assignment", "no_fragment_edges"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base.set_seed(args.seed)
    global_data = torch.load(args.global_cache, map_location="cpu", weights_only=False)
    drug_data = torch.load(args.drug_cache, map_location="cpu", weights_only=False)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    # Entity-level pretraining uses one representative pair row per training drug.
    # Validation and test rows are never materialized.
    unique_rows, seen = [], set()
    for row in split["train_indices"]:
        drug_id = str(global_data["drug_id"][row])
        if drug_id not in seen:
            unique_rows.append(int(row))
            seen.add(drug_id)
    store = ChemStore(global_data, drug_data, args.condition, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        base.IndexDataset(unique_rows), batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0, collate_fn=store.collate,
    )
    model = BRICSChemHierarchicalP13D(args.project, args.checkpoint).to(args.device)
    prefixes = (
        "fragment_input", "fragment_blocks", "mask_token",
        "fragment_reconstruction", "fragment_alignment",
    )
    selected = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    ]
    optimizer = torch.optim.AdamW(selected, lr=args.lr, weight_decay=1e-5)

    iterator = iter(loader)
    running = {"loss": 0.0, "reconstruction": 0.0, "alignment": 0.0}
    history = []
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = base.move(batch, args.device)
        optimizer.zero_grad(set_to_none=True)
        _, debug = model(batch, use_mask=True, return_debug=True)
        loss = debug["reconstruction_loss"] + 0.1 * debug["alignment_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selected, 5.0)
        optimizer.step()
        running["loss"] += float(loss.detach())
        running["reconstruction"] += float(debug["reconstruction_loss"].detach())
        running["alignment"] += float(debug["alignment_loss"].detach())
        if step % 100 == 0 or step == args.steps:
            row = {"step": step, **{key: value / 100 for key, value in running.items()}}
            history.append(row)
            print(json.dumps(row), flush=True)
            running = {"loss": 0.0, "reconstruction": 0.0, "alignment": 0.0}

    selected_names = {
        name for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    }
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in selected_names
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "guardrail": "unique training drugs only; validation/test rows and labels were not used",
        "condition": args.condition,
        "seed": args.seed,
        "training_drugs": len(unique_rows),
        "steps": args.steps,
        "history": history,
    }
    torch.save({"hierarchy_state": state, "result": result}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
