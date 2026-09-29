#!/usr/bin/env python3
"""Train clean BRICS graph with joint drug/protein similarity smoothing."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_brics_hierarchical_stage1 as base
from train_brics_chem_stage1 import ChemStore
from train_brics_graph_stage1 import BRICSChemGraphP13D


class PairSimilarityStore(ChemStore):
    def __init__(self, global_data, drug_data, similarity_data, condition, seed):
        super().__init__(global_data, drug_data, "real", seed)
        self.protein_ids = [str(x) for x in global_data["protein_id"]]
        self.drug_similarity = similarity_data["drug_similarity"].float()
        self.protein_similarity = similarity_data["protein_similarity"].float()
        drug_lookup = {x: i for i, x in enumerate(similarity_data["drug_ids"])}
        protein_lookup = {x: i for i, x in enumerate(similarity_data["protein_ids"])}
        self.drug_index = {x: drug_lookup[x] for x in set(self.drug_ids)}
        self.protein_index = {x: protein_lookup[x] for x in set(self.protein_ids)}
        if condition == "shuffled_joint":
            generator = torch.Generator().manual_seed(seed + 12091)
            drug_keys = sorted(self.drug_index)
            protein_keys = sorted(self.protein_index)
            drug_values = [self.drug_index[x] for x in drug_keys]
            protein_values = [self.protein_index[x] for x in protein_keys]
            dp = torch.randperm(len(drug_keys), generator=generator).tolist()
            pp = torch.randperm(len(protein_keys), generator=generator).tolist()
            self.drug_index = {x: drug_values[j] for x, j in zip(drug_keys, dp)}
            self.protein_index = {x: protein_values[j] for x, j in zip(protein_keys, pp)}

    def collate(self, rows):
        output = super().collate(rows)
        di = torch.tensor([self.drug_index[self.drug_ids[row]] for row in rows], dtype=torch.long)
        pi = torch.tensor([self.protein_index[self.protein_ids[row]] for row in rows], dtype=torch.long)
        output["batch_drug_similarity"] = self.drug_similarity[di][:, di]
        output["batch_protein_similarity"] = self.protein_similarity[pi][:, pi]
        return output


class PretrainedPairSmoothBRICS(BRICSChemGraphP13D):
    def __init__(self, project, checkpoint, pretrained):
        super().__init__(project, checkpoint)
        payload = torch.load(pretrained, map_location="cpu", weights_only=False)
        if payload["result"]["condition"] != "real":
            raise ValueError("pair smoothing requires real BRICS pretraining")
        incompatible = self.load_state_dict(payload["hierarchy_state"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected pretrained keys: {incompatible.unexpected_keys}")


def joint_pair_loss(prediction, baseline, drug_similarity, protein_similarity, drug_floor, protein_floor):
    drug_weight = torch.relu((drug_similarity - drug_floor) / (1.0 - drug_floor))
    protein_weight = torch.relu((protein_similarity - protein_floor) / (1.0 - protein_floor))
    weight = drug_weight * protein_weight
    upper = torch.triu(torch.ones_like(weight, dtype=torch.bool), diagonal=1)
    weight = weight[upper]
    active = weight > 0
    if not active.any():
        zero = prediction.new_zeros(())
        return zero, zero, zero
    correction = prediction - baseline
    difference = (correction[:, None] - correction[None, :]).pow(2)[upper]
    loss = (weight[active] * difference[active]).sum() / weight[active].sum().clamp_min(1e-8)
    return loss, active.float().mean(), weight[active].mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/data1/ztx/MyModel-MDTA"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained-hierarchy", type=Path, required=True)
    parser.add_argument("--clean-reference-predictions", type=Path, required=True)
    parser.add_argument("--global-cache", type=Path, required=True)
    parser.add_argument("--drug-cache", type=Path, required=True)
    parser.add_argument("--similarity-cache", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--condition", choices=["real", "shuffled_joint"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-align", type=float, default=0.05)
    parser.add_argument("--lambda-reconstruct", type=float, default=0.02)
    parser.add_argument("--lambda-delta", type=float, default=0.001)
    parser.add_argument("--lambda-pair", type=float, required=True)
    parser.add_argument("--drug-floor", type=float, default=0.15)
    parser.add_argument("--protein-floor", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    sys.path.insert(0, str(args.project.resolve()))
    from experiments.klifs85_interaction.train_klifs_interact import metrics

    global_data = torch.load(args.global_cache, map_location="cpu", weights_only=False)
    drug_data = torch.load(args.drug_cache, map_location="cpu", weights_only=False)
    similarity_data = torch.load(args.similarity_cache, map_location="cpu", weights_only=False)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_indices, val_indices = split["train_indices"], split["val_indices"]
    store = PairSimilarityStore(global_data, drug_data, similarity_data, args.condition, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        base.IndexDataset(train_indices), batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0, collate_fn=store.collate,
    )
    val_loader = DataLoader(
        base.IndexDataset(val_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=store.collate,
    )
    model = PretrainedPairSmoothBRICS(
        args.project, args.checkpoint, args.pretrained_hierarchy
    ).to(args.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    baseline = store.prediction[val_indices].numpy()
    labels = store.label[val_indices].numpy()
    baseline_metrics = metrics(labels, baseline)
    clean_reference = np.load(args.clean_reference_predictions)
    if not np.array_equal(clean_reference["indices"], np.asarray(val_indices)):
        raise RuntimeError("clean reference indices do not match")
    clean_prediction = clean_reference["prediction"]
    clean_metrics = metrics(labels, clean_prediction)
    initial = base.evaluate(model, val_loader, args.device)
    initial_error = float(torch.max(torch.abs(initial["prediction"] - torch.from_numpy(baseline))))
    if initial_error > 1e-5:
        raise RuntimeError(f"epoch-0 mismatch versus P13D: {initial_error}")

    best_mse, best_epoch, no_improve = baseline_metrics["mse"], 0, 0
    best_state = base.trainable_state(model)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0, "mse": 0.0, "align": 0.0, "reconstruct": 0.0,
            "pair": 0.0, "active_fraction": 0.0, "active_weight": 0.0, "count": 0,
        }
        for batch in train_loader:
            batch = base.move(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction, debug = model(batch, use_mask=True, return_debug=True)
            mse = F.mse_loss(prediction, batch["label"])
            pair, active_fraction, active_weight = joint_pair_loss(
                prediction, batch["baseline_prediction"],
                batch["batch_drug_similarity"], batch["batch_protein_similarity"],
                args.drug_floor, args.protein_floor,
            )
            loss = (
                mse + args.lambda_align * debug["alignment_loss"]
                + args.lambda_reconstruct * debug["reconstruction_loss"]
                + args.lambda_delta * debug["delta"].pow(2).mean()
                + args.lambda_pair * pair
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            count = len(prediction)
            for key, value in [
                ("loss", loss), ("mse", mse), ("align", debug["alignment_loss"]),
                ("reconstruct", debug["reconstruction_loss"]), ("pair", pair),
                ("active_fraction", active_fraction), ("active_weight", active_weight),
            ]:
                totals[key] += float(value.detach()) * count
            totals["count"] += count

        validation = base.evaluate(model, val_loader, args.device)
        val_metrics = metrics(validation["label"].numpy(), validation["prediction"].numpy())
        row = {
            "epoch": epoch,
            "train": {k: totals[k] / totals["count"] for k in totals if k != "count"},
            "val": val_metrics,
            "val_delta_abs_mean": float(validation["delta"].abs().mean()),
            "val_gate_mean": float(validation["gate"].mean()),
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
    best = base.evaluate(model, val_loader, args.device)
    best_prediction = best["prediction"].numpy()
    result = {
        "guardrail": "validation only; test indices and metrics were not accessed",
        "condition": args.condition, "seed": args.seed, "best_epoch": best_epoch,
        "epoch0_max_abs_difference_vs_p13d": initial_error,
        "p13d_baseline": baseline_metrics, "clean_brics_reference": clean_metrics,
        "best": metrics(labels, best_prediction),
        "relative_mse_gain_vs_p13d": float((baseline_metrics["mse"] - best_mse) / baseline_metrics["mse"]),
        "relative_mse_gain_vs_clean_brics": float((clean_metrics["mse"] - best_mse) / clean_metrics["mse"]),
        "drug_bootstrap_vs_clean_brics": base.bootstrap_by_drug(
            val_indices, store.drug_ids, clean_prediction, best_prediction, labels,
            seed=20260903 + args.seed,
        ),
        "drug_bootstrap_vs_p13d": base.bootstrap_by_drug(
            val_indices, store.drug_ids, baseline, best_prediction, labels
        ),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"trainable_state": best_state, "result": result}, args.output_dir / "best.pt")
    np.savez_compressed(
        args.output_dir / "validation_predictions.npz", indices=np.asarray(val_indices), labels=labels,
        p13d=baseline, clean=clean_prediction, prediction=best_prediction,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

