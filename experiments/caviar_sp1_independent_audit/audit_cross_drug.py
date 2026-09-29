from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.collate_p13d_caviar_subpocket import (
    mdta_collate_fn_p13d_caviar_subpocket,
    move_batch_to_device_caviar_subpocket,
)
from datasets.davis_dataset_p13d_caviar_subpocket import (
    DavisDatasetP13DCaviarSubpocket,
)
from experiments.caviar_sp1_independent_audit.audit_checkpoint import (
    Intervention,
    set_seed,
)
from models.model_p13d_caviar_subpocket import CaviarSubpocketDTA
from train_p13d_caviar_subpocket import metrics


def protein_major_indices(dataset, indices):
    """Put different drugs for the same protein next to each other."""
    return sorted(
        indices,
        key=lambda index: (
            str(dataset.df.iloc[index]["protein_id"]),
            str(dataset.df.iloc[index]["drug_id"]),
        ),
    )


@torch.no_grad()
def evaluate(model, loader, device, mode, prediction_floor):
    model.eval()
    predictions, targets, drug_ids = [], [], []
    with Intervention(model, mode) as intervention:
        for batch in loader:
            intervention.set_batch(batch["drug_id"])
            batch = move_batch_to_device_caviar_subpocket(batch, device, True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(batch, return_details=True)["pred"].float()
            prediction = prediction.clamp_min(prediction_floor)
            predictions.append(prediction.view(-1).cpu())
            targets.append(batch["label"].float().view(-1).cpu())
            drug_ids.extend(batch["drug_id"])
    prediction = torch.cat(predictions).numpy()
    target = torch.cat(targets).numpy()
    return metrics(target, prediction, drug_ids), prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=7)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    split = json.loads(Path(config["split_json"]).read_text())
    device = torch.device(args.device)
    set_seed(config["seed"])

    dataset = DavisDatasetP13DCaviarSubpocket(
        pairs_csv=config["pairs_csv"],
        drug_1d_dir=config["drug_1d_dir"],
        protein_1d_dir=config["protein_1d_dir"],
        protein_3d_dir=config["protein_3d_dir"],
        use_drug_2d=False,
        use_drug_3d=False,
        drug_atom_v2_dir=config["drug_atom_v2_dir"],
        protein_residue_v2_dir=config["protein_residue_v2_dir"],
        protein_subpocket_dir=config["protein_subpocket_dir"],
    )
    collate = partial(
        mdta_collate_fn_p13d_caviar_subpocket,
        max_fragments=config["max_fragments"],
        max_atoms_per_fragment=config["max_atoms_per_fragment"],
        max_subpockets=config["max_subpockets"],
        max_residues_per_subpocket=config["max_residues_per_subpocket"],
    )
    model = CaviarSubpocketDTA(
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        subpocket_rounds=config["subpocket_rounds"],
        region_rounds=config["region_rounds"],
        pockets_per_fragment=config["pockets_per_fragment"],
        delta_max=config["delta_max"],
        ar_gate_epsilon=config["ar_gate_epsilon"],
        fp_gate_max=config["fp_gate_max"],
    ).to(device)
    checkpoint = torch.load(
        run_dir / "checkpoint_best_val_mse.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    report = {
        "audit": "CAVIAR_SP1_CROSS_DRUG_AR_EVIDENCE_PERMUTATION",
        "ordering": "protein-major; each batch contains different drugs for one protein",
        "batch_size": args.batch_size,
        "no_training_performed": True,
        "splits": {},
    }
    for split_name in ("val", "test"):
        ordered = protein_major_indices(dataset, split[f"{split_name}_indices"])
        loader = DataLoader(
            Subset(dataset, ordered),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate,
            pin_memory=True,
            persistent_workers=True,
        )
        real_metrics, real_prediction = evaluate(
            model, loader, device, "real", config["prediction_floor"]
        )
        perm_metrics, perm_prediction = evaluate(
            model,
            loader,
            device,
            "ar_evidence_all_permute",
            config["prediction_floor"],
        )
        change = perm_prediction - real_prediction
        report["splits"][split_name] = {
            "real": real_metrics,
            "cross_drug_permuted_ar": perm_metrics,
            "delta_mse": perm_metrics["mse"] - real_metrics["mse"],
            "mean_abs_prediction_change": float(np.mean(np.abs(change))),
            "max_abs_prediction_change": float(np.max(np.abs(change))),
            "prediction_correlation": float(
                np.corrcoef(real_prediction, perm_prediction)[0, 1]
            ),
        }
        print(
            f"{split_name.upper()} REAL={real_metrics['mse']:.9f} "
            f"CROSS_DRUG_PERM={perm_metrics['mse']:.9f} "
            f"DELTA={perm_metrics['mse'] - real_metrics['mse']:+.9f}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
