#!/usr/bin/env python3
"""Stage-1 causal test of BRICS-fragment/KLIFS-pocket dual graphs over frozen P13D."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairDataset(Dataset):
    def __init__(self, indices):
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return int(self.indices[item])


def graph_data(item, self_loop=False, protein=False):
    edge_index, edge_attr = item["edge_index"], item["edge_attr"]
    if self_loop:
        keep = edge_index[0] == edge_index[1]
        edge_index, edge_attr = edge_index[:, keep], edge_attr[keep]
    kwargs = {"x": item["x"].float(), "edge_index": edge_index.long(), "edge_attr": edge_attr.float()}
    if protein:
        kwargs["valid_mask"] = item["mask"].bool()
        # Avoid the substring "index": PyG automatically offsets attributes
        # named like indices while batching multiple graphs.
        kwargs["klifs_pos_id"] = item["klifs_index"].long()
    return Data(**kwargs)


class Store:
    def __init__(self, global_data, entity_data, condition, seed):
        self.global_feature = global_data["pair_feature"].float()
        self.frozen_prediction = global_data["prediction"].float()
        self.label = global_data["label"].float()
        self.drug_ids = [str(x) for x in global_data["drug_id"]]
        self.protein_ids = [str(x) for x in global_data["protein_id"]]
        self.drugs = {
            key: graph_data(value, self_loop=condition == "self_loop")
            for key, value in entity_data["drugs"].items()
        }
        proteins = entity_data["proteins"]
        accepted = sorted(key for key, value in proteins.items() if bool(value["mask"].any()))
        invalid = sorted(set(proteins) - set(accepted))
        mapping = {key: key for key in proteins}
        if condition == "shuffled":
            rng = np.random.default_rng(seed)
            shuffled = list(rng.permutation(accepted))
            mapping.update(dict(zip(accepted, shuffled)))
            mapping.update({key: key for key in invalid})
        self.proteins = {
            key: graph_data(proteins[mapping[key]], self_loop=condition == "self_loop", protein=True)
            for key in proteins
        }

    def collate(self, indices):
        indices = torch.tensor(indices, dtype=torch.long)
        rows = indices.tolist()
        return {
            "index": indices,
            "global_feature": self.global_feature[indices],
            "frozen_prediction": self.frozen_prediction[indices],
            "label": self.label[indices],
            "drug_graph": Batch.from_data_list([self.drugs[self.drug_ids[i]] for i in rows]),
            "protein_graph": Batch.from_data_list([self.proteins[self.protein_ids[i]] for i in rows]),
        }


class EdgeGATBlock(nn.Module):
    def __init__(self, hidden, edge_dim, heads=4, dropout=0.1):
        super().__init__()
        self.conv = GATv2Conv(
            hidden, hidden // heads, heads=heads, concat=True,
            edge_dim=edge_dim, dropout=dropout, add_self_loops=False,
        )
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        return self.norm(x + self.dropout(F.silu(self.conv(x, edge_index, edge_attr))))


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Linear(hidden, 1)

    def forward(self, values, mask):
        score = self.score(values).squeeze(-1).masked_fill(~mask, -1e4)
        weight = torch.softmax(score, dim=-1) * mask.float()
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
        return torch.sum(weight.unsqueeze(-1) * values, dim=1)


class DualGraphStage1(nn.Module):
    def __init__(self, hidden=128, dropout=0.1):
        super().__init__()
        self.drug_input = nn.Sequential(nn.Linear(256, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.protein_input = nn.Sequential(nn.Linear(1152, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.klifs_position = nn.Embedding(85, hidden)
        self.drug_blocks = nn.ModuleList([EdgeGATBlock(hidden, 7, dropout=dropout) for _ in range(2)])
        self.protein_blocks = nn.ModuleList([EdgeGATBlock(hidden, 19, dropout=dropout) for _ in range(2)])
        self.fragment_from_pocket = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
        self.pocket_from_fragment = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
        self.fragment_norm = nn.LayerNorm(hidden)
        self.pocket_norm = nn.LayerNorm(hidden)
        self.fragment_pool = MaskedAttentionPool(hidden)
        self.pocket_pool = MaskedAttentionPool(hidden)
        self.local_projection = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(256 + hidden, 256), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.SiLU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, batch, return_debug=False):
        drug = batch["drug_graph"]
        protein = batch["protein_graph"]
        fragment = self.drug_input(drug.x)
        for block in self.drug_blocks:
            fragment = block(fragment, drug.edge_index, drug.edge_attr)
        pocket = self.protein_input(protein.x) + self.klifs_position(protein.klifs_pos_id)
        pocket = pocket * protein.valid_mask.unsqueeze(-1)
        for block in self.protein_blocks:
            pocket = block(pocket, protein.edge_index, protein.edge_attr)
            pocket = pocket * protein.valid_mask.unsqueeze(-1)
        fragment, fragment_mask = to_dense_batch(fragment, drug.batch)
        pocket, pocket_batch_mask = to_dense_batch(pocket, protein.batch)
        valid, _ = to_dense_batch(protein.valid_mask.float(), protein.batch)
        pocket_mask = pocket_batch_mask & valid.bool()
        reliability = pocket_mask.any(dim=1)
        safe_pocket_mask = pocket_mask.clone()
        safe_pocket_mask[~reliability, 0] = True
        pocket = pocket * pocket_mask.unsqueeze(-1)
        fragment_context = self.fragment_from_pocket(
            fragment, pocket, pocket, key_padding_mask=~safe_pocket_mask, need_weights=False
        )[0]
        pocket_context = self.pocket_from_fragment(
            pocket, fragment, fragment, key_padding_mask=~fragment_mask, need_weights=False
        )[0]
        fragment = self.fragment_norm(fragment + fragment_context)
        pocket = self.pocket_norm(pocket + pocket_context) * pocket_mask.unsqueeze(-1)
        fragment_summary = self.fragment_pool(fragment, fragment_mask)
        pocket_summary = self.pocket_pool(pocket, pocket_mask)
        local = self.local_projection(
            torch.cat([fragment_summary, pocket_summary, fragment_summary * pocket_summary], dim=-1)
        )
        local = local * reliability.unsqueeze(-1)
        correction = self.residual_head(torch.cat([batch["global_feature"], local], dim=-1)).squeeze(-1)
        prediction = batch["frozen_prediction"] + correction
        if return_debug:
            return prediction, correction, reliability
        return prediction


def move(batch, device):
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, labels, indices, corrections, reliability = [], [], [], [], []
    for batch in loader:
        batch = move(batch, device)
        prediction, correction, reliable = model(batch, return_debug=True)
        predictions.append(prediction.cpu())
        labels.append(batch["label"].cpu())
        indices.append(batch["index"].cpu())
        corrections.append(correction.cpu())
        reliability.append(reliable.cpu())
    return {
        "prediction": torch.cat(predictions), "label": torch.cat(labels), "index": torch.cat(indices),
        "correction": torch.cat(corrections), "reliability": torch.cat(reliability),
    }


def bootstrap_by_drug(rows, drug_ids, baseline, prediction, label, n=20000, seed=20260901):
    groups = {}
    for local, row in enumerate(rows):
        groups.setdefault(drug_ids[int(row)], []).append(local)
    improvements = []
    for positions in groups.values():
        positions = np.asarray(positions, dtype=int)
        improvements.append(float(np.mean((baseline[positions] - label[positions]) ** 2 - (prediction[positions] - label[positions]) ** 2)))
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
    parser.add_argument("--global-cache", type=Path, required=True)
    parser.add_argument("--entity-cache", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--condition", choices=["real", "shuffled", "self_loop"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    set_seed(args.seed)
    sys.path.insert(0, str(args.project.resolve()))
    from experiments.klifs85_interaction.train_klifs_interact import metrics

    global_data = torch.load(args.global_cache, map_location="cpu", weights_only=False)
    entity_data = torch.load(args.entity_cache, map_location="cpu", weights_only=False)
    split = json.loads(args.split.read_text())
    # The test split is intentionally never materialized.
    train_indices, val_indices = split["train_indices"], split["val_indices"]
    store = Store(global_data, entity_data, args.condition, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        PairDataset(train_indices), batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=0, collate_fn=store.collate,
    )
    val_loader = DataLoader(
        PairDataset(val_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=store.collate,
    )
    model = DualGraphStage1().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    baseline = store.frozen_prediction[val_indices].numpy()
    labels = store.label[val_indices].numpy()
    baseline_metrics = metrics(labels, baseline)
    best_mse, best_epoch, no_improve = baseline_metrics["mse"], 0, 0
    best_state, history = None, []
    initial = evaluate(model, val_loader, args.device)
    initial_error = float(torch.max(torch.abs(initial["prediction"] - initial["label"].new_tensor(baseline))))
    if initial_error > 1e-6:
        raise RuntimeError(f"epoch-0 prediction mismatch: {initial_error}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            batch = move(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = F.mse_loss(prediction, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss) * len(prediction)
            count += len(prediction)
        validation = evaluate(model, val_loader, args.device)
        val_metrics = metrics(validation["label"].numpy(), validation["prediction"].numpy())
        history.append({"epoch": epoch, "train_mse": total / count, "val": val_metrics})
        print(json.dumps(history[-1]), flush=True)
        if val_metrics["mse"] < best_mse - 1e-6:
            best_mse, best_epoch, no_improve = val_metrics["mse"], epoch, 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    best = evaluate(model, val_loader, args.device) if best_state is not None else initial
    best_prediction = best["prediction"].numpy() if best_state is not None else baseline
    result = {
        "guardrail": "validation only; test indices and metrics were not accessed",
        "condition": args.condition, "seed": args.seed, "best_epoch": best_epoch,
        "epoch0_max_abs_difference": initial_error,
        "baseline": baseline_metrics, "best": metrics(labels, best_prediction),
        "relative_mse_gain": float((baseline_metrics["mse"] - best_mse) / baseline_metrics["mse"]),
        "correction_abs_mean": float(best["correction"].abs().mean()),
        "local_reliability_mean": float(best["reliability"].float().mean()),
        "drug_bootstrap": bootstrap_by_drug(val_indices, store.drug_ids, baseline, best_prediction, labels),
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": best_state, "result": result}, args.output_dir / "best.pt")
    np.savez_compressed(
        args.output_dir / "validation_predictions.npz", indices=np.asarray(val_indices), labels=labels,
        baseline=baseline, prediction=best_prediction,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
