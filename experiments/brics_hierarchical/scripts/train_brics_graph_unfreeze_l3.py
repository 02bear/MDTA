#!/usr/bin/env python3
"""Jointly fine-tune a clean BRICS graph branch and only drug EGNN layer 3."""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_brics_hierarchical_stage1 as base
from train_brics_chem_stage1 import ChemStore
from train_brics_graph_stage1 import BRICSChemGraphP13D


class Layer3Store(ChemStore):
    def __init__(self, global_data, drug_data, layer2_data, condition, seed):
        super().__init__(global_data, drug_data, condition, seed)
        self.layer2 = layer2_data["drugs"]
        missing = sorted(set(self.drugs) - set(self.layer2))
        if missing:
            raise KeyError(f"layer-2 cache is missing drugs: {missing[:5]}")

    def collate(self, rows):
        output = super().collate(rows)
        hs, positions, batches, edges = [], [], [], []
        offset = 0
        for batch_id, row in enumerate(rows):
            drug_id = self.drug_ids[row]
            item = self.layer2[drug_id]
            count = item["h"].shape[0]
            hs.append(item["h"].float())
            positions.append(item["pos"].float())
            batches.append(torch.full((count,), batch_id, dtype=torch.long))
            edges.append(item["edge_index"].long() + offset)
            offset += count
        output.update({
            "drug_l2_h": torch.cat(hs),
            "drug_l2_pos": torch.cat(positions),
            "drug_l2_batch": torch.cat(batches),
            "drug_l2_edge_index": torch.cat(edges, dim=1),
        })
        return output


class BRICSGraphUnfreezeL3(BRICSChemGraphP13D):
    def __init__(self, project, checkpoint, clean_checkpoint, condition, hidden=128, dropout=0.1):
        super().__init__(project, checkpoint, hidden, dropout)
        clean = torch.load(clean_checkpoint, map_location="cpu", weights_only=False)
        saved_condition = clean["result"]["condition"]
        if saved_condition != condition:
            raise ValueError(f"clean checkpoint condition={saved_condition}, requested={condition}")
        incompatible = self.load_state_dict(clean["hierarchy_state"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected clean checkpoint keys: {incompatible.unexpected_keys}")

        sys.path.insert(0, str(project.resolve()))
        from models.model_p13d import MyModelMDTAP13D
        base_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
        ckpt_args = base_checkpoint.get("args", {})
        p13d = MyModelMDTAP13D(
            drug_1d_in_dim=int(ckpt_args.get("drug_1d_in_dim", 768)),
            drug_3d_node_in_dim=int(ckpt_args.get("drug_3d_node_in_dim", 10)),
            protein_1d_in_dim=1280,
            protein_3d_node_s_dim=6,
            protein_3d_node_v_dim=3,
            hidden_dim=int(ckpt_args.get("hidden_dim", 128)),
            dropout=float(ckpt_args.get("dropout", 0.1)),
            task="regression",
        )
        p13d.load_state_dict(base_checkpoint["model_state_dict"], strict=True)
        self.drug_3d_layer3 = copy.deepcopy(p13d.drug_3d_encoder.layers[2])
        self.drug_3d_out_proj = copy.deepcopy(p13d.drug_3d_encoder.out_proj)
        for parameter in self.drug_3d_out_proj.parameters():
            parameter.requires_grad = False
        for module in (self.broadcast, self.atom_gate):
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.clean_result = clean["result"]

    def train(self, mode=True):
        super().train(mode)
        self.drug_3d_out_proj.eval()
        return self

    def forward(self, batch, use_mask=False, return_debug=False):
        chemistry = batch["fragment_chem_feat"]
        fragment_count = batch["fragment_batch"].numel()
        fragment = self.fragment_input(chemistry)
        masked = torch.zeros(fragment_count, dtype=torch.bool, device=fragment.device)
        if use_mask and fragment_count:
            masked = torch.rand(fragment_count, device=fragment.device) < self.mask_rate
            if not masked.any():
                masked[torch.randint(fragment_count, (1,), device=fragment.device)] = True
            fragment = torch.where(masked[:, None], self.mask_token[None, :], fragment)
        for block in self.fragment_blocks:
            fragment = block(fragment, batch["fragment_edge_index"], batch["fragment_edge_attr"])
        fragment_global = global_mean_pool(fragment, batch["fragment_batch"])

        h3, _ = self.drug_3d_layer3(
            batch["drug_l2_h"], batch["drug_l2_pos"], batch["drug_l2_edge_index"]
        )
        drug_3d_base = self.drug_3d_out_proj(global_mean_pool(h3, batch["drug_l2_batch"]))
        gate = torch.sigmoid(self.graph_gate(torch.cat([drug_3d_base, fragment_global], dim=-1)))
        raw_delta = self.delta_3d(torch.cat(
            [drug_3d_base, fragment_global, drug_3d_base * fragment_global], dim=-1
        ))
        delta = gate * raw_delta
        drug_3d = drug_3d_base + delta
        drug_fused = self.drug_fusion([batch["drug_1d_feat"], drug_3d])
        pair_feature = torch.cat([drug_fused, batch["pair_feature"][:, 128:]], dim=-1)
        prediction = self.decoder(pair_feature).view(-1)

        align = 1.0 - F.cosine_similarity(
            self.fragment_alignment(fragment_global), drug_3d_base.detach(), dim=-1
        ).mean()
        if masked.any():
            reconstructed = self.fragment_reconstruction(fragment[masked])
            target = chemistry[masked].detach()
            reconstruction = (
                F.binary_cross_entropy_with_logits(reconstructed[:, :272], target[:, :272])
                + 0.2 * F.mse_loss(reconstructed[:, 272:], target[:, 272:])
            )
        else:
            reconstruction = prediction.new_zeros(())
        debug = {
            "delta": delta,
            "gate": gate,
            "alignment_loss": align,
            "reconstruction_loss": reconstruction,
            "drug_3d_base": drug_3d_base,
        }
        return (prediction, debug) if return_debug else prediction


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
    parser.add_argument("--layer2-cache", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--condition", choices=["real", "no_fragment_edges"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--branch-lr", type=float, default=1e-4)
    parser.add_argument("--layer3-lr", type=float, default=1e-5)
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
    layer2_data = torch.load(args.layer2_cache, map_location="cpu", weights_only=False)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_indices, val_indices = split["train_indices"], split["val_indices"]
    store = Layer3Store(global_data, drug_data, layer2_data, args.condition, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        base.IndexDataset(train_indices), batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0, collate_fn=store.collate,
    )
    val_loader = DataLoader(
        base.IndexDataset(val_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=store.collate,
    )
    model = BRICSGraphUnfreezeL3(
        args.project, args.checkpoint, args.clean_checkpoint, args.condition
    ).to(args.device)
    layer3_params = [p for p in model.drug_3d_layer3.parameters() if p.requires_grad]
    branch_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("drug_3d_layer3.")
    ]
    optimizer = torch.optim.AdamW([
        {"params": branch_params, "lr": args.branch_lr},
        {"params": layer3_params, "lr": args.layer3_lr},
    ], weight_decay=args.weight_decay)

    p13d_baseline = store.prediction[val_indices].numpy()
    labels = store.label[val_indices].numpy()
    reference = np.load(args.reference_predictions)
    if not np.array_equal(reference["indices"], np.asarray(val_indices)):
        raise RuntimeError("reference validation indices do not match current split")
    if float(np.max(np.abs(reference["labels"] - labels))) > 1e-7:
        raise RuntimeError("reference validation labels do not match current split")
    frozen_prediction = reference["prediction"]
    frozen_metrics = metrics(labels, frozen_prediction)
    initial = base.evaluate(model, val_loader, args.device)
    initial_error = float(np.max(np.abs(initial["prediction"].numpy() - frozen_prediction)))
    if initial_error > 2e-5:
        raise RuntimeError(f"epoch-0 prediction mismatch versus clean frozen model: {initial_error}")

    best_mse, best_epoch, no_improve = frozen_metrics["mse"], 0, 0
    best_state = base.trainable_state(model)
    history = []
    trainable_counts = {
        "branch": int(sum(p.numel() for p in branch_params)),
        "drug_egnn_layer3": int(sum(p.numel() for p in layer3_params)),
    }
    print(json.dumps({
        "epoch0_max_abs_difference_vs_frozen": initial_error,
        "frozen_metrics": frozen_metrics,
        "trainable_parameters": trainable_counts,
    }), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "mse": 0.0, "align": 0.0, "reconstruct": 0.0, "count": 0}
        for batch in train_loader:
            batch = base.move(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction, debug = model(batch, use_mask=True, return_debug=True)
            mse = F.mse_loss(prediction, batch["label"])
            loss = (
                mse + args.lambda_align * debug["alignment_loss"]
                + args.lambda_reconstruct * debug["reconstruction_loss"]
                + args.lambda_delta * debug["delta"].pow(2).mean()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(branch_params + layer3_params, 5.0)
            optimizer.step()
            count = len(prediction)
            totals["loss"] += float(loss.detach()) * count
            totals["mse"] += float(mse.detach()) * count
            totals["align"] += float(debug["alignment_loss"].detach()) * count
            totals["reconstruct"] += float(debug["reconstruction_loss"].detach()) * count
            totals["count"] += count

        validation = base.evaluate(model, val_loader, args.device)
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
            best_state = base.trainable_state(model)
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break

    model.load_state_dict(best_state, strict=False)
    best = base.evaluate(model, val_loader, args.device)
    best_prediction = best["prediction"].numpy()
    p13d_metrics = metrics(labels, p13d_baseline)
    result = {
        "guardrail": "validation only; test indices and metrics were not accessed",
        "condition": args.condition,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "epoch0_max_abs_difference_vs_frozen": initial_error,
        "p13d_baseline": p13d_metrics,
        "frozen_brics_baseline": frozen_metrics,
        "best": metrics(labels, best_prediction),
        "relative_mse_gain_vs_frozen": float((frozen_metrics["mse"] - best_mse) / frozen_metrics["mse"]),
        "relative_mse_gain_vs_p13d": float((p13d_metrics["mse"] - best_mse) / p13d_metrics["mse"]),
        "delta_abs_mean": float(best["delta"].abs().mean()),
        "graph_gate_mean": float(best["gate"].mean()),
        "drug_bootstrap_vs_p13d": base.bootstrap_by_drug(
            val_indices, store.drug_ids, p13d_baseline, best_prediction, labels
        ),
        "drug_bootstrap_vs_frozen": base.bootstrap_by_drug(
            val_indices, store.drug_ids, frozen_prediction, best_prediction, labels,
            seed=20260901 + args.seed,
        ),
        "trainable_parameters": trainable_counts,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"trainable_state": best_state, "result": result}, args.output_dir / "best.pt")
    np.savez_compressed(
        args.output_dir / "validation_predictions.npz",
        indices=np.asarray(val_indices), labels=labels, p13d=p13d_baseline,
        frozen=frozen_prediction, prediction=best_prediction,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

