#!/usr/bin/env python3
"""Calibrate a frozen clean BRICS residual using label-free reliability features."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_brics_hierarchical_stage1 as base
from train_brics_chem_stage1 import ChemStore
from train_brics_graph_stage1 import BRICSChemGraphP13D


class ReliabilityStore(ChemStore):
    def __init__(self, global_data, drug_data, reliability_data, condition, seed):
        super().__init__(global_data, drug_data, "real", seed)
        feature_map = reliability_data["features"]
        self.reliability = {x: feature_map[x].float().clone() for x in self.drugs}
        if condition == "shuffled_reliability":
            ids = sorted(self.reliability)
            generator = torch.Generator().manual_seed(seed + 9103)
            permutation = torch.randperm(len(ids), generator=generator).tolist()
            original = {x: self.reliability[x] for x in ids}
            self.reliability = {x: original[ids[j]].clone() for x, j in zip(ids, permutation)}
        elif condition == "constant_reliability":
            self.reliability = {x: torch.zeros_like(value) for x, value in self.reliability.items()}

    def collate(self, rows):
        output = super().collate(rows)
        output["reliability_feat"] = torch.stack([
            self.reliability[self.drug_ids[row]] for row in rows
        ])
        return output


class ReliabilityGatedBRICS(BRICSChemGraphP13D):
    def __init__(self, project, checkpoint, clean_checkpoint, clean_condition, max_adjustment=0.25):
        super().__init__(project, checkpoint)
        clean = torch.load(clean_checkpoint, map_location="cpu", weights_only=False)
        if clean["result"]["condition"] != clean_condition:
            raise ValueError("clean checkpoint condition mismatch")
        incompatible = self.load_state_dict(clean["hierarchy_state"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected clean checkpoint keys: {incompatible.unexpected_keys}")
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.reliability_head = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.reliability_head[-1].weight)
        nn.init.zeros_(self.reliability_head[-1].bias)
        self.max_adjustment = max_adjustment

    def train(self, mode=True):
        nn.Module.train(self, False)
        self.reliability_head.train(mode)
        return self

    def forward(self, batch, use_mask=False, return_debug=False):
        chemistry = batch["fragment_chem_feat"]
        fragment = self.fragment_input(chemistry)
        for block in self.fragment_blocks:
            fragment = block(fragment, batch["fragment_edge_index"], batch["fragment_edge_attr"])
        fragment_global = global_mean_pool(fragment, batch["fragment_batch"])
        baseline_3d = batch["drug_3d_graph_feat"]
        base_gate = torch.sigmoid(self.graph_gate(torch.cat([baseline_3d, fragment_global], dim=-1)))
        multiplier = 1.0 + self.max_adjustment * torch.tanh(
            self.reliability_head(batch["reliability_feat"])
        )
        raw_delta = self.delta_3d(torch.cat(
            [baseline_3d, fragment_global, baseline_3d * fragment_global], dim=-1
        ))
        delta = base_gate * multiplier * raw_delta
        drug_3d = baseline_3d + delta
        drug_fused = self.drug_fusion([batch["drug_1d_feat"], drug_3d])
        pair_feature = torch.cat([drug_fused, batch["pair_feature"][:, 128:]], dim=-1)
        prediction = self.decoder(pair_feature).view(-1)
        debug = {"delta": delta, "gate": base_gate, "multiplier": multiplier}
        return (prediction, debug) if return_debug else prediction


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, labels, indices, deltas, gates, multipliers = [], [], [], [], [], []
    for batch in loader:
        batch = base.move(batch, device)
        prediction, debug = model(batch, return_debug=True)
        predictions.append(prediction.cpu())
        labels.append(batch["label"].cpu())
        indices.append(batch["index"].cpu())
        deltas.append(debug["delta"].cpu())
        gates.append(debug["gate"].cpu())
        multipliers.append(debug["multiplier"].cpu())
    return {
        "prediction": torch.cat(predictions), "label": torch.cat(labels), "index": torch.cat(indices),
        "delta": torch.cat(deltas), "gate": torch.cat(gates), "multiplier": torch.cat(multipliers),
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/data1/ztx/MyModel-MDTA"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clean-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--global-cache", type=Path, required=True)
    parser.add_argument("--drug-cache", type=Path, required=True)
    parser.add_argument("--reliability-cache", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--condition",
        choices=["real", "shuffled_reliability", "constant_reliability"],
        required=True,
    )
    parser.add_argument("--clean-condition", choices=["real", "no_fragment_edges"], default="real")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-adjustment", type=float, default=0.25)
    parser.add_argument("--lambda-adjustment", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    sys.path.insert(0, str(args.project.resolve()))
    from experiments.klifs85_interaction.train_klifs_interact import metrics

    global_data = torch.load(args.global_cache, map_location="cpu", weights_only=False)
    drug_data = torch.load(args.drug_cache, map_location="cpu", weights_only=False)
    reliability_data = torch.load(args.reliability_cache, map_location="cpu", weights_only=False)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_indices, val_indices = split["train_indices"], split["val_indices"]
    store = ReliabilityStore(global_data, drug_data, reliability_data, args.condition, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        base.IndexDataset(train_indices), batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0, collate_fn=store.collate,
    )
    val_loader = DataLoader(
        base.IndexDataset(val_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=store.collate,
    )
    model = ReliabilityGatedBRICS(
        args.project, args.checkpoint, args.clean_checkpoint,
        args.clean_condition, args.max_adjustment,
    ).to(args.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    labels = store.label[val_indices].numpy()
    p13d_prediction = store.prediction[val_indices].numpy()
    reference = np.load(args.reference_predictions)
    if not np.array_equal(reference["indices"], np.asarray(val_indices)):
        raise RuntimeError("reference validation indices do not match")
    frozen_prediction = reference["prediction"]
    frozen_metrics = metrics(labels, frozen_prediction)
    initial = evaluate(model, val_loader, args.device)
    initial_error = float(np.max(np.abs(initial["prediction"].numpy() - frozen_prediction)))
    if initial_error > 2e-5:
        raise RuntimeError(f"epoch-0 mismatch versus frozen BRICS: {initial_error}")

    best_mse, best_epoch, no_improve = frozen_metrics["mse"], 0, 0
    best_state = base.trainable_state(model)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "mse": 0.0, "adjustment": 0.0, "count": 0}
        for batch in train_loader:
            batch = base.move(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction, debug = model(batch, return_debug=True)
            mse = F.mse_loss(prediction, batch["label"])
            adjustment = (debug["multiplier"] - 1.0).pow(2).mean()
            loss = mse + args.lambda_adjustment * adjustment
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            count = len(prediction)
            totals["loss"] += float(loss.detach()) * count
            totals["mse"] += float(mse.detach()) * count
            totals["adjustment"] += float(adjustment.detach()) * count
            totals["count"] += count
        validation = evaluate(model, val_loader, args.device)
        val_metrics = metrics(validation["label"].numpy(), validation["prediction"].numpy())
        row = {
            "epoch": epoch,
            "train": {k: totals[k] / totals["count"] for k in ["loss", "mse", "adjustment"]},
            "val": val_metrics,
            "val_multiplier_mean": float(validation["multiplier"].mean()),
            "val_multiplier_min": float(validation["multiplier"].min()),
            "val_multiplier_max": float(validation["multiplier"].max()),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["mse"] < best_mse - 1e-6:
            best_mse, best_epoch, no_improve = val_metrics["mse"], epoch, 0
            best_state = base.trainable_state(model)
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break

    model.load_state_dict(best_state, strict=False)
    best = evaluate(model, val_loader, args.device)
    best_prediction = best["prediction"].numpy()
    p13d_metrics = metrics(labels, p13d_prediction)
    result = {
        "guardrail": "validation only; test indices and metrics were not accessed",
        "condition": args.condition, "seed": args.seed, "best_epoch": best_epoch,
        "epoch0_max_abs_difference_vs_frozen": initial_error,
        "p13d_baseline": p13d_metrics, "frozen_brics_baseline": frozen_metrics,
        "best": metrics(labels, best_prediction),
        "relative_mse_gain_vs_frozen": float((frozen_metrics["mse"] - best_mse) / frozen_metrics["mse"]),
        "relative_mse_gain_vs_p13d": float((p13d_metrics["mse"] - best_mse) / p13d_metrics["mse"]),
        "multiplier": {
            "mean": float(best["multiplier"].mean()), "std": float(best["multiplier"].std()),
            "min": float(best["multiplier"].min()), "max": float(best["multiplier"].max()),
        },
        "drug_bootstrap_vs_frozen": base.bootstrap_by_drug(
            val_indices, store.drug_ids, frozen_prediction, best_prediction, labels,
            seed=20260902 + args.seed,
        ),
        "drug_bootstrap_vs_p13d": base.bootstrap_by_drug(
            val_indices, store.drug_ids, p13d_prediction, best_prediction, labels
        ),
        "trainable_parameters": int(sum(p.numel() for p in trainable)),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"trainable_state": best_state, "result": result}, args.output_dir / "best.pt")
    np.savez_compressed(
        args.output_dir / "validation_predictions.npz", indices=np.asarray(val_indices), labels=labels,
        p13d=p13d_prediction, frozen=frozen_prediction, prediction=best_prediction,
        multiplier=best["multiplier"].numpy(),
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
