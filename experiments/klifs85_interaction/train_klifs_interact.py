#!/usr/bin/env python3
"""Protected P13D + Davis-native KLIFS atom-residue interaction experiments.

K1 trains the local branch with affinity loss only.  K2 adds a weak soft
contrastive objective whose targets are ECFP4 * KLIFS85 similarities.  Epoch 0
is the exact frozen P13D validation baseline and remains selectable.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FenwickTree:
    def __init__(self, n):
        self.n, self.tree = n, np.zeros(n + 1, dtype=np.int64)

    def update(self, i):
        while i <= self.n:
            self.tree[i] += 1
            i += i & -i

    def query(self, i):
        value = 0
        while i > 0:
            value += self.tree[i]
            i -= i & -i
        return value


def cindex(y, p):
    ranks = {value: i + 1 for i, value in enumerate(np.unique(p))}
    order = np.argsort(y, kind="mergesort")
    y, p = y[order], p[order]
    tree, previous, concordant, comparable, start = FenwickTree(len(ranks)), 0, 0.0, 0.0, 0
    while start < len(y):
        end = start
        while end < len(y) and y[end] == y[start]:
            end += 1
        for k in range(start, end):
            rank = ranks[p[k]]
            less = tree.query(rank - 1)
            equal = tree.query(rank) - less
            concordant += less + 0.5 * equal
            comparable += previous
        for k in range(start, end):
            tree.update(ranks[p[k]])
            previous += 1
        start = end
    return float(concordant / comparable) if comparable else 0.0


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def rm2(y, p):
    yc, pc = y - y.mean(), p - p.mean()
    denom = np.sum(yc * yc) * np.sum(pc * pc)
    r2 = float(np.sum(yc * pc) ** 2 / denom) if denom else 0.0
    pdenom = np.sum(p * p)
    k = np.sum(y * p) / pdenom if pdenom else 0.0
    ydenom = np.sum((y - y.mean()) ** 2)
    r02 = float(1 - np.sum((y - k * p) ** 2) / ydenom) if ydenom else 0.0
    return float(r2 * (1 - math.sqrt(abs(r2**2 - r02**2))))


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    error = p - y
    mse = float(np.mean(error**2))
    yr, pr = rankdata(y), rankdata(p)
    denom = np.sum((y - y.mean()) ** 2)
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "ci": cindex(y, p),
        "rm2": rm2(y, p),
        "pearson": float(np.corrcoef(y, p)[0, 1]) if np.std(y) and np.std(p) else 0.0,
        "spearman": float(np.corrcoef(yr, pr)[0, 1]) if np.std(yr) and np.std(pr) else 0.0,
        "r2": float(1 - np.sum(error**2) / denom) if denom else 0.0,
        "bias": float(np.mean(error)),
    }


class IndexDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index


class JointNeighborBatchSampler(Sampler):
    def __init__(self, indices, drug_index, protein_index, train_drugs, drug_sim, protein_sim, batch_size, seed):
        self.indices = np.asarray(indices, dtype=int)
        self.drug_index = np.asarray(drug_index, dtype=int)
        self.protein_index = np.asarray(protein_index, dtype=int)
        self.train_drugs = np.asarray(train_drugs, dtype=int)
        self.drug_sim, self.protein_sim = drug_sim, protein_sim
        self.anchor_batch = max(1, batch_size // 2)
        self.seed, self.epoch = seed, 0
        self.grid = {(self.drug_index[i], self.protein_index[i]): int(i) for i in self.indices}
        self.neighbors = self._build_neighbors()

    def _build_neighbors(self):
        output = {}
        for index in self.indices:
            d, p = self.drug_index[index], self.protein_index[index]
            ds = self.train_drugs[np.argsort(self.drug_sim[d, self.train_drugs])[::-1][:12]]
            ps = np.argsort(self.protein_sim[p])[::-1][:32]
            candidates = []
            for nd in ds:
                for np_ in ps:
                    neighbor = self.grid.get((int(nd), int(np_)))
                    if neighbor is None or neighbor == index:
                        continue
                    score = float(self.drug_sim[d, nd] * self.protein_sim[p, np_])
                    if score >= 0.60:
                        candidates.append((neighbor, max(1e-8, ((score - 0.60) / 0.40) ** 2)))
            if not candidates:
                candidates = [(int(index), 1.0)]
            ids = np.asarray([x[0] for x in candidates], dtype=int)
            probs = np.asarray([x[1] for x in candidates], dtype=float)
            output[int(index)] = (ids, probs / probs.sum())
        return output

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        anchors = rng.permutation(self.indices)
        for start in range(0, len(anchors), self.anchor_batch):
            chunk = anchors[start : start + self.anchor_batch]
            partners = []
            for index in chunk:
                ids, probabilities = self.neighbors[int(index)]
                partners.append(rng.choice(ids, p=probabilities))
            # Every train row is an affinity anchor exactly once per epoch;
            # sampled partners are present only to form meaningful soft pairs.
            yield [(int(index), True) for index in chunk] + [(int(index), False) for index in partners]

    def __len__(self):
        return math.ceil(len(self.indices) / self.anchor_batch)


class PairStore:
    def __init__(self, global_data, feature_data, ligand_dir, similarity_path):
        self.prediction = global_data["prediction"].float()
        self.label = global_data["label"].float()
        self.drug_ids = [str(x) for x in global_data["drug_id"]]
        self.protein_ids = [str(x) for x in global_data["protein_id"]]
        sim = np.load(similarity_path)
        sim_drugs = [str(x) for x in sim["drug_ids"].tolist()]
        sim_proteins = [str(x) for x in sim["protein_ids"].tolist()]
        self.drug_lookup = {x: i for i, x in enumerate(sim_drugs)}
        self.protein_lookup = {x: i for i, x in enumerate(sim_proteins)}
        self.drug_index = np.asarray([self.drug_lookup[x] for x in self.drug_ids], dtype=int)
        self.protein_index = np.asarray([self.protein_lookup[x] for x in self.protein_ids], dtype=int)
        self.drug_similarity = sim["drug_similarity"].astype(np.float32)
        self.protein_similarity = sim["protein_similarity"].astype(np.float32)
        self.features = feature_data["proteins"]
        self.graphs = {}
        for drug_id in sorted(set(self.drug_ids)):
            item = torch.load(ligand_dir / f"{drug_id}.pt", map_location="cpu", weights_only=False)
            self.graphs[drug_id] = Data(
                x=item["x"].float(),
                edge_index=item["edge_index"].long(),
                edge_attr=item["edge_attr"].float(),
            )

    def collate(self, indices):
        if indices and isinstance(indices[0], tuple):
            anchor_mask = torch.tensor([item[1] for item in indices], dtype=torch.bool)
            indices = [item[0] for item in indices]
        else:
            anchor_mask = torch.ones(len(indices), dtype=torch.bool)
        indices = torch.tensor(indices, dtype=torch.long)
        drug_ids = [self.drug_ids[i] for i in indices.tolist()]
        protein_ids = [self.protein_ids[i] for i in indices.tolist()]
        proteins = [self.features[x] for x in protein_ids]
        return {
            "indices": indices,
            "anchor_mask": anchor_mask,
            "graphs": Batch.from_data_list([self.graphs[x] for x in drug_ids]),
            "sequence": torch.stack([x["sequence"] for x in proteins]).float(),
            "geometry": torch.stack([x["geometry"] for x in proteins]).float(),
            "mask": torch.stack([x["mask"] for x in proteins]),
            "frozen_prediction": self.prediction[indices],
            "label": self.label[indices],
            "drug_index": torch.tensor(self.drug_index[indices], dtype=torch.long),
            "protein_index": torch.tensor(self.protein_index[indices], dtype=torch.long),
        }


class KLIFSInteractP13D(nn.Module):
    def __init__(self, hidden=128, dropout=0.1):
        super().__init__()
        self.atom_input = nn.Sequential(nn.Linear(97, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.atom_convs = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden)), edge_dim=6),
            GINEConv(nn.Sequential(nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden)), edge_dim=6),
        ])
        seq_layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout, batch_first=True, norm_first=True)
        geo_layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout, batch_first=True, norm_first=True)
        self.sequence_input = nn.Sequential(nn.Linear(1024, hidden), nn.LayerNorm(hidden))
        self.geometry_input = nn.Sequential(nn.Linear(18, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.position = nn.Parameter(torch.randn(1, 85, hidden) * 0.02)
        self.sequence_encoder = nn.TransformerEncoder(seq_layer, 1)
        self.geometry_encoder = nn.TransformerEncoder(geo_layer, 1)
        self.fusion_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.atom_q = nn.Linear(hidden, hidden, bias=False)
        self.residue_k = nn.Linear(hidden, hidden, bias=False)
        self.local_projection = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden)
        )
        self.gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.5)
        self.residual = nn.Sequential(nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, 1, bias=False))
        nn.init.zeros_(self.residual[-1].weight)

    def forward(self, batch):
        graph = batch["graphs"]
        atoms = self.atom_input(graph.x)
        for conv in self.atom_convs:
            atoms = F.silu(conv(atoms, graph.edge_index, graph.edge_attr) + atoms)
        atoms, atom_mask = to_dense_batch(atoms, graph.batch)

        mask = batch["mask"]
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True
        sequence = self.sequence_input(batch["sequence"])
        geometry = self.geometry_input(batch["geometry"]) + self.position
        sequence = self.sequence_encoder(sequence, src_key_padding_mask=~safe_mask)
        geometry = self.geometry_encoder(geometry, src_key_padding_mask=~safe_mask)
        sequence = sequence * mask.unsqueeze(-1)
        geometry = geometry * mask.unsqueeze(-1)
        fusion = self.fusion_gate(torch.cat([sequence, geometry], dim=-1))
        residues = fusion * sequence + (1.0 - fusion) * geometry

        logits = torch.einsum("bah,brh->bar", self.atom_q(atoms), self.residue_k(residues)) / math.sqrt(atoms.shape[-1])
        pair_mask = atom_mask.unsqueeze(2) & mask.unsqueeze(1)
        safe_pair_mask = pair_mask.clone()
        safe_pair_mask[empty, 0, 0] = True
        attention = torch.softmax(logits.masked_fill(~safe_pair_mask, -1e4).flatten(1), dim=-1).reshape_as(logits)
        attention = attention * pair_mask
        atom_pool = torch.einsum("bar,bah->bh", attention, atoms)
        residue_pool = torch.einsum("bar,brh->bh", attention, residues)
        local = self.local_projection(torch.cat([atom_pool, residue_pool, atom_pool * residue_pool, (atom_pool - residue_pool).abs()], dim=-1))
        reliability = mask.float().mean(dim=1)
        local = local * (reliability > 0).float().unsqueeze(-1)
        gate = torch.sigmoid(self.gate(local)).squeeze(-1) * reliability
        delta = gate * 2.0 * torch.tanh(self.residual(local).squeeze(-1))
        prediction = batch["frozen_prediction"] + delta
        return {"prediction": prediction, "local": local, "attention": attention, "gate": gate, "delta": delta}


def soft_joint_contrast(local, drug_index, protein_index, drug_similarity, protein_similarity, temperature=0.15):
    z = F.normalize(local, dim=-1)
    logits = z @ z.T / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    joint = drug_similarity[drug_index[:, None], drug_index[None, :]] * protein_similarity[
        protein_index[:, None], protein_index[None, :]
    ]
    weights = ((joint - 0.60).clamp_min(0.0) / 0.40).square().masked_fill(eye, 0.0)
    valid = weights.sum(dim=1) > 1e-8
    if not valid.any():
        return local.sum() * 0.0
    targets = weights[valid] / weights[valid].sum(dim=1, keepdim=True)
    log_probs = F.log_softmax(logits[valid].masked_fill(eye[valid], -1e4), dim=-1)
    return -(targets * log_probs).sum(dim=1).mean()


def move(batch, device):
    return {key: (value.to(device) if hasattr(value, "to") else value) for key, value in batch.items()}


def evaluate(model, loader, device):
    model.eval()
    predictions, labels, gates, deltas = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            output = model(move(batch, device))
            predictions.append(output["prediction"].cpu())
            labels.append(batch["label"])
            gates.append(output["gate"].cpu())
            deltas.append(output["delta"].cpu())
    return torch.cat(predictions), torch.cat(labels), torch.cat(gates), torch.cat(deltas)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def fit(condition, seed, store, train_idx, val_idx, train_drugs, args, baseline):
    set_seed(seed)
    dataset = IndexDataset(len(store.label))
    sampler = JointNeighborBatchSampler(
        train_idx, store.drug_index, store.protein_index, train_drugs,
        store.drug_similarity, store.protein_similarity, args.batch_size, seed,
    )
    train_loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=store.collate, num_workers=0)
    val_loader = DataLoader(dataset, batch_size=args.eval_batch_size, sampler=val_idx.tolist(), collate_fn=store.collate, num_workers=0)
    model = KLIFSInteractP13D(hidden=args.hidden, dropout=args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    drug_similarity = torch.tensor(store.drug_similarity, device=args.device)
    protein_similarity = torch.tensor(store.protein_similarity, device=args.device)
    sim_weight = 0.0 if condition == "K1" else args.sim_weight

    best_mse, best_state, best_epoch, stale = baseline["mse"], None, 0, 0
    history = [{"epoch": 0, "val_mse": baseline["mse"], "enabled": False}]
    for epoch in range(1, args.epochs + 1):
        model.train()
        affinity_sum = contrast_sum = gate_sum = 0.0
        batches = 0
        for batch in train_loader:
            batch = move(batch, args.device)
            output = model(batch)
            affinity = F.mse_loss(
                output["prediction"][batch["anchor_mask"]], batch["label"][batch["anchor_mask"]]
            )
            contrast = soft_joint_contrast(
                output["local"], batch["drug_index"], batch["protein_index"], drug_similarity, protein_similarity
            ) if sim_weight else output["local"].sum() * 0.0
            loss = affinity + sim_weight * contrast
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            affinity_sum += float(affinity.detach())
            contrast_sum += float(contrast.detach())
            gate_sum += float(output["gate"].mean().detach())
            batches += 1
            if args.smoke_batches and batches >= args.smoke_batches:
                break
        prediction, label, gates, deltas = evaluate(model, val_loader, args.device)
        val_mse = float((prediction - label).square().mean())
        history.append({
            "epoch": epoch,
            "val_mse": val_mse,
            "train_affinity_mse": affinity_sum / batches,
            "train_soft_contrast": contrast_sum / batches,
            "train_gate_mean": gate_sum / batches,
            "val_gate_mean": float(gates.mean()),
            "val_delta_std": float(deltas.std()),
            "enabled": True,
        })
        print(json.dumps({"condition": condition, "seed": seed, **history[-1]}), flush=True)
        if val_mse < best_mse - args.minimum_improvement:
            best_mse, best_state, best_epoch, stale = val_mse, cpu_state(model), epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    enabled = best_state is not None
    if enabled:
        model.load_state_dict(best_state)
        prediction, label, gates, deltas = evaluate(model, val_loader, args.device)
    else:
        prediction = store.prediction[val_idx].clone()
        label = store.label[val_idx].clone()
        gates = torch.zeros_like(prediction)
        deltas = torch.zeros_like(prediction)
    result = metrics(label.numpy(), prediction.numpy())
    result.update({
        "condition": condition,
        "seed": seed,
        "enabled": enabled,
        "best_epoch": best_epoch,
        "absolute_mse_improvement": baseline["mse"] - result["mse"],
        "relative_mse_improvement": (baseline["mse"] - result["mse"]) / baseline["mse"],
        "gate_mean": float(gates.mean()),
        "delta_std": float(deltas.std()),
        "history": history,
    })
    return result, {"enabled": enabled, "state_dict": best_state, "condition": condition, "seed": seed, "best_epoch": best_epoch}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--ligand-dir", type=Path, required=True)
    parser.add_argument("--similarity", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=["K1", "K2"], choices=["K1", "K2"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sim-weight", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--minimum-improvement", type=float, default=1e-5)
    parser.add_argument("--smoke-batches", type=int, default=0, help="Limit train batches for an implementation smoke test")
    args = parser.parse_args()

    global_data = torch.load(args.global_cache, map_location="cpu", weights_only=False)
    feature_data = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    store = PairStore(global_data, feature_data, args.ligand_dir, args.similarity)
    split = json.loads(args.split.read_text())
    train_idx = np.asarray(split["train_indices"], dtype=int)
    val_idx = np.asarray(split["val_indices"], dtype=int)
    # Intentionally never materialize or score split['test_indices'].
    train_drugs = np.asarray([store.drug_lookup[str(x)] for x in split["train_drugs"]], dtype=int)
    baseline = metrics(store.label[val_idx].numpy(), store.prediction[val_idx].numpy())
    baseline.update({"validation_rows": len(val_idx), "test_rows_accessed": 0})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"global_frozen": baseline, "runs": []}
    for condition in args.conditions:
        for seed in args.seeds:
            result, checkpoint = fit(condition, seed, store, train_idx, val_idx, train_drugs, args, baseline)
            report["runs"].append(result)
            torch.save(checkpoint, args.output_dir / f"{condition}_seed{seed}.pt")
            print(json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)
            (args.output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    grouped = {}
    for condition in args.conditions:
        rows = [row for row in report["runs"] if row["condition"] == condition]
        grouped[condition] = {
            "mse_mean": float(np.mean([row["mse"] for row in rows])),
            "mse_std": float(np.std([row["mse"] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
            "relative_improvement_mean": float(np.mean([row["relative_mse_improvement"] for row in rows])),
            "enabled_seeds": int(sum(row["enabled"] for row in rows)),
        }
    report["summary"] = grouped
    report["guardrails"] = {
        "epoch_zero_exact_frozen_baseline": True,
        "test_rows_accessed": 0,
        "low_quality_klifs_targets_masked": True,
    }
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"global_frozen": baseline, "summary": grouped, "guardrails": report["guardrails"]}, indent=2))


if __name__ == "__main__":
    main()
