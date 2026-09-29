#!/usr/bin/env python3
"""Causal screen for BRICS hierarchy inside the frozen P13D drug encoder."""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import scatter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IndexDataset(Dataset):
    def __init__(self, indices):
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return int(self.indices[index])


class Store:
    def __init__(self, global_data, drug_data, condition, seed):
        self.prediction = global_data["prediction"].float()
        self.pair_feature = global_data["pair_feature"].float()
        self.label = global_data["label"].float()
        self.drug_ids = [str(x) for x in global_data["drug_id"]]
        self.drugs = drug_data["drugs"]
        self.condition = condition
        self.assignment = {}
        generator = torch.Generator().manual_seed(seed + 1729)
        for drug_id in sorted(self.drugs):
            original = self.drugs[drug_id]["atom_to_fragment"].long()
            if condition == "random_assignment":
                self.assignment[drug_id] = original[torch.randperm(len(original), generator=generator)]
            else:
                self.assignment[drug_id] = original

    def collate(self, rows):
        atom_nodes, atom_batch, atom_to_fragment = [], [], []
        fragment_batch, edge_indices, edge_attrs = [], [], []
        atom_offset = 0
        fragment_offset = 0
        for batch_id, row in enumerate(rows):
            drug_id = self.drug_ids[row]
            item = self.drugs[drug_id]
            atoms = item["atom_node_feat"].float()
            assignment = self.assignment[drug_id]
            n_fragments = int(item["n_fragments"])
            if len(atoms) != len(assignment):
                raise ValueError(f"{drug_id}: atom/mapping length mismatch")
            atom_nodes.append(atoms)
            atom_batch.append(torch.full((len(atoms),), batch_id, dtype=torch.long))
            atom_to_fragment.append(assignment + fragment_offset)
            fragment_batch.append(torch.full((n_fragments,), batch_id, dtype=torch.long))
            edge_index = item["fragment_edge_index"].long()
            edge_attr = item["fragment_edge_attr"].float()
            if self.condition == "no_fragment_edges":
                keep = edge_index[0] == edge_index[1]
                edge_index, edge_attr = edge_index[:, keep], edge_attr[keep]
            edge_indices.append(edge_index + fragment_offset)
            edge_attrs.append(edge_attr)
            atom_offset += len(atoms)
            fragment_offset += n_fragments
        index = torch.tensor(rows, dtype=torch.long)
        return {
            "index": index,
            "pair_feature": self.pair_feature[index],
            "baseline_prediction": self.prediction[index],
            "label": self.label[index],
            "drug_1d_feat": torch.stack([self.drugs[self.drug_ids[r]]["drug_1d_feat"] for r in rows]),
            "drug_3d_graph_feat": torch.stack([self.drugs[self.drug_ids[r]]["drug_3d_graph_feat"] for r in rows]),
            "atom_node_feat": torch.cat(atom_nodes),
            "atom_batch": torch.cat(atom_batch),
            "atom_to_fragment": torch.cat(atom_to_fragment),
            "fragment_batch": torch.cat(fragment_batch),
            "fragment_edge_index": torch.cat(edge_indices, dim=1),
            "fragment_edge_attr": torch.cat(edge_attrs),
        }


class EdgeGATBlock(nn.Module):
    def __init__(self, hidden=128, edge_dim=7, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.conv = GATv2Conv(
            hidden, hidden // 4, heads=4, concat=True, edge_dim=edge_dim,
            dropout=dropout, add_self_loops=False,
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.Dropout(dropout),
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.norm1(x + self.conv(x, edge_index, edge_attr))
        return self.norm2(x + self.ffn(x))


class BRICSHierarchicalP13D(nn.Module):
    def __init__(self, project, checkpoint, hidden=128, dropout=0.1, mask_rate=0.15):
        super().__init__()
        sys.path.insert(0, str(project.resolve()))
        from models.model_p13d import MyModelMDTAP13D

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {})
        base = MyModelMDTAP13D(
            drug_1d_in_dim=int(ckpt_args.get("drug_1d_in_dim", 768)),
            drug_3d_node_in_dim=int(ckpt_args.get("drug_3d_node_in_dim", 10)),
            protein_1d_in_dim=1280,
            protein_3d_node_s_dim=6,
            protein_3d_node_v_dim=3,
            hidden_dim=int(ckpt_args.get("hidden_dim", 128)),
            dropout=float(ckpt_args.get("dropout", 0.1)),
            task="regression",
        )
        base.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.drug_fusion = base.drug_fusion
        self.decoder = base.decoder
        for parameter in self.drug_fusion.parameters():
            parameter.requires_grad = False
        for parameter in self.decoder.parameters():
            parameter.requires_grad = False

        self.fragment_input = nn.Sequential(nn.Linear(256, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.fragment_blocks = nn.ModuleList([EdgeGATBlock(hidden, 7, dropout) for _ in range(2)])
        self.mask_token = nn.Parameter(torch.zeros(hidden))
        self.fragment_reconstruction = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, 256)
        )
        self.fragment_alignment = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 128))
        self.broadcast = nn.Linear(hidden, 128)
        self.atom_gate = nn.Linear(256, 1)
        self.delta_3d = nn.Sequential(
            nn.Linear(128 * 3, 256), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
        )
        nn.init.zeros_(self.delta_3d[-1].weight)
        nn.init.zeros_(self.delta_3d[-1].bias)
        self.mask_rate = mask_rate

    def train(self, mode=True):
        super().train(mode)
        self.drug_fusion.eval()
        self.decoder.eval()
        return self

    def forward(self, batch, use_mask=False, return_debug=False):
        atoms = batch["atom_node_feat"]
        assignment = batch["atom_to_fragment"]
        fragment_count = batch["fragment_batch"].numel()
        fragment_mean = scatter(atoms, assignment, dim=0, dim_size=fragment_count, reduce="mean")
        fragment_max = scatter(atoms, assignment, dim=0, dim_size=fragment_count, reduce="max")
        raw_fragment = torch.cat([fragment_mean, fragment_max], dim=-1)
        fragment = self.fragment_input(raw_fragment)

        masked = torch.zeros(fragment_count, dtype=torch.bool, device=fragment.device)
        if use_mask and fragment_count:
            masked = torch.rand(fragment_count, device=fragment.device) < self.mask_rate
            if not masked.any():
                masked[torch.randint(fragment_count, (1,), device=fragment.device)] = True
            fragment = torch.where(masked[:, None], self.mask_token[None, :], fragment)
        for block in self.fragment_blocks:
            fragment = block(fragment, batch["fragment_edge_index"], batch["fragment_edge_attr"])

        fragment_global = global_mean_pool(fragment, batch["fragment_batch"])
        broadcast = self.broadcast(fragment[assignment])
        gate = torch.sigmoid(self.atom_gate(torch.cat([atoms, broadcast], dim=-1)))
        updated_atoms = atoms + gate * broadcast
        updated_graph = global_mean_pool(updated_atoms, batch["atom_batch"])
        baseline_3d = batch["drug_3d_graph_feat"]
        delta = self.delta_3d(torch.cat([baseline_3d, updated_graph, updated_graph - baseline_3d], dim=-1))
        drug_3d = baseline_3d + delta
        drug_fused = self.drug_fusion([batch["drug_1d_feat"], drug_3d])
        pair_feature = torch.cat([drug_fused, batch["pair_feature"][:, 128:]], dim=-1)
        prediction = self.decoder(pair_feature).view(-1)

        align = 1.0 - F.cosine_similarity(
            self.fragment_alignment(fragment_global), baseline_3d.detach(), dim=-1
        ).mean()
        if masked.any():
            reconstruction = F.smooth_l1_loss(
                self.fragment_reconstruction(fragment[masked]), raw_fragment[masked].detach()
            )
        else:
            reconstruction = prediction.new_zeros(())
        debug = {
            "delta": delta,
            "gate": gate,
            "alignment_loss": align,
            "reconstruction_loss": reconstruction,
        }
        return (prediction, debug) if return_debug else prediction


def move(batch, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, labels, indices, deltas, gates = [], [], [], [], []
    for batch in loader:
        batch = move(batch, device)
        prediction, debug = model(batch, use_mask=False, return_debug=True)
        predictions.append(prediction.cpu())
        labels.append(batch["label"].cpu())
        indices.append(batch["index"].cpu())
        deltas.append(debug["delta"].cpu())
        gates.append(debug["gate"].cpu())
    return {
        "prediction": torch.cat(predictions), "label": torch.cat(labels),
        "index": torch.cat(indices), "delta": torch.cat(deltas), "gate": torch.cat(gates),
    }


def trainable_state(model):
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in names}


def bootstrap_by_drug(rows, drug_ids, baseline, prediction, label, n=20000, seed=20260901):
    groups = {}
    for local, row in enumerate(rows):
        groups.setdefault(drug_ids[int(row)], []).append(local)
    improvements = []
    for positions in groups.values():
        positions = np.asarray(positions, dtype=int)
        improvements.append(float(np.mean(
            (baseline[positions] - label[positions]) ** 2
            - (prediction[positions] - label[positions]) ** 2
        )))
    improvements = np.asarray(improvements)
    rng = np.random.default_rng(seed)
    samples = improvements[rng.integers(0, len(improvements), size=(n, len(improvements)))].mean(1)
    return {
        "drugs": len(improvements), "improved": int(np.sum(improvements > 0)),
        "worsened": int(np.sum(improvements < 0)), "improvements": improvements.tolist(),
        "mean": float(improvements.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "probability_positive": float(np.mean(samples > 0)),
    }


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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-align", type=float, default=0.05)
    parser.add_argument("--lambda-reconstruct", type=float, default=0.02)
    parser.add_argument("--lambda-delta", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    sys.path.insert(0, str(args.project.resolve()))
    from experiments.klifs85_interaction.train_klifs_interact import metrics

    global_data = torch.load(args.global_cache, map_location="cpu", weights_only=False)
    drug_data = torch.load(args.drug_cache, map_location="cpu", weights_only=False)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_indices, val_indices = split["train_indices"], split["val_indices"]
    store = Store(global_data, drug_data, args.condition, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        IndexDataset(train_indices), batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0, collate_fn=store.collate,
    )
    val_loader = DataLoader(
        IndexDataset(val_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=store.collate,
    )
    model = BRICSHierarchicalP13D(args.project, args.checkpoint).to(args.device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    baseline = store.prediction[val_indices].numpy()
    labels = store.label[val_indices].numpy()
    baseline_metrics = metrics(labels, baseline)
    initial = evaluate(model, val_loader, args.device)
    initial_error = float(torch.max(torch.abs(initial["prediction"] - torch.from_numpy(baseline))))
    if initial_error > 1e-5:
        raise RuntimeError(f"epoch-0 prediction mismatch: {initial_error}")

    best_mse, best_epoch, no_improve = baseline_metrics["mse"], 0, 0
    best_state = trainable_state(model)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "mse": 0.0, "align": 0.0, "reconstruct": 0.0, "count": 0}
        for batch in train_loader:
            batch = move(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction, debug = model(batch, use_mask=True, return_debug=True)
            mse = F.mse_loss(prediction, batch["label"])
            delta_penalty = debug["delta"].pow(2).mean()
            loss = (
                mse + args.lambda_align * debug["alignment_loss"]
                + args.lambda_reconstruct * debug["reconstruction_loss"]
                + args.lambda_delta * delta_penalty
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            count = len(prediction)
            totals["loss"] += float(loss.detach()) * count
            totals["mse"] += float(mse.detach()) * count
            totals["align"] += float(debug["alignment_loss"].detach()) * count
            totals["reconstruct"] += float(debug["reconstruction_loss"].detach()) * count
            totals["count"] += count

        validation = evaluate(model, val_loader, args.device)
        val_metrics = metrics(validation["label"].numpy(), validation["prediction"].numpy())
        row = {
            "epoch": epoch,
            "train": {k: totals[k] / totals["count"] for k in ["loss", "mse", "align", "reconstruct"]},
            "val": val_metrics,
            "val_delta_abs_mean": float(validation["delta"].abs().mean()),
            "val_gate_mean": float(validation["gate"].mean()),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["mse"] < best_mse - 1e-6:
            best_mse, best_epoch, no_improve = val_metrics["mse"], epoch, 0
            best_state = trainable_state(model)
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break

    model.load_state_dict(best_state, strict=False)
    best = evaluate(model, val_loader, args.device)
    best_prediction = best["prediction"].numpy()
    result = {
        "guardrail": "validation only; test indices and metrics were not accessed",
        "condition": args.condition, "seed": args.seed, "best_epoch": best_epoch,
        "epoch0_max_abs_difference": initial_error,
        "baseline": baseline_metrics,
        "best": metrics(labels, best_prediction),
        "relative_mse_gain": float((baseline_metrics["mse"] - best_mse) / baseline_metrics["mse"]),
        "delta_abs_mean": float(best["delta"].abs().mean()),
        "atom_gate_mean": float(best["gate"].mean()),
        "drug_bootstrap": bootstrap_by_drug(val_indices, store.drug_ids, baseline, best_prediction, labels),
        "args": vars(args), "history": history,
    }
    result["args"] = {k: str(v) if isinstance(v, Path) else v for k, v in result["args"].items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"hierarchy_state": best_state, "result": result}, args.output_dir / "best.pt")
    np.savez_compressed(
        args.output_dir / "validation_predictions.npz",
        indices=np.asarray(val_indices), labels=labels, baseline=baseline, prediction=best_prediction,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
