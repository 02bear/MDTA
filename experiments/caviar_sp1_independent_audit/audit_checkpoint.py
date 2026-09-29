from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import ExitStack
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
from models.model_p13d_caviar_subpocket import CaviarSubpocketDTA
from train_p13d_caviar_subpocket import metrics


MODES = (
    "real",
    "global_base_only",
    "no_fp_delta",
    "no_ar_delta",
    "ar_evidence_zero",
    "ar_evidence_same_drug_permute",
    "ar_evidence_all_permute",
    "ar_head_dp_only",
    "ar_head_local_only",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_same_drug_permutation(drug_ids: list[str], device: torch.device) -> torch.Tensor:
    permutation = torch.arange(len(drug_ids), device=device)
    grouped: dict[str, list[int]] = {}
    for index, drug_id in enumerate(drug_ids):
        grouped.setdefault(str(drug_id), []).append(index)
    for indices in grouped.values():
        if len(indices) > 1:
            source = indices[-1:] + indices[:-1]
            permutation[torch.as_tensor(indices, device=device)] = torch.as_tensor(
                source, device=device
            )
    return permutation


class Intervention:
    def __init__(self, model: CaviarSubpocketDTA, mode: str):
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.model = model
        self.mode = mode
        self.drug_ids: list[str] = []
        self.handles = []

    def set_batch(self, drug_ids: list[str]) -> None:
        self.drug_ids = [str(value) for value in drug_ids]

    def _edit_ar_input(self, _module, inputs):
        (features,) = inputs
        hidden = features.shape[-1] // 3
        drug_protein = features[:, : 2 * hidden]
        local = features[:, 2 * hidden :]
        if self.mode in {"ar_evidence_zero", "ar_head_dp_only"}:
            local = torch.zeros_like(local)
        elif self.mode == "ar_evidence_same_drug_permute":
            permutation = make_same_drug_permutation(self.drug_ids, features.device)
            local = local[permutation]
        elif self.mode == "ar_evidence_all_permute":
            permutation = torch.roll(
                torch.arange(features.shape[0], device=features.device), shifts=1
            )
            local = local[permutation]
        elif self.mode == "ar_head_local_only":
            drug_protein = torch.zeros_like(drug_protein)
        return (torch.cat([drug_protein, local], dim=-1),)

    @staticmethod
    def _zero_output(_module, _inputs, output):
        return torch.zeros_like(output)

    def __enter__(self):
        if self.mode in {
            "ar_evidence_zero",
            "ar_evidence_same_drug_permute",
            "ar_evidence_all_permute",
            "ar_head_dp_only",
            "ar_head_local_only",
        }:
            self.handles.extend(
                [
                    self.model.ar_delta_head.register_forward_pre_hook(
                        self._edit_ar_input
                    ),
                    self.model.ar_gate.register_forward_pre_hook(self._edit_ar_input),
                ]
            )
        if self.mode in {"no_ar_delta", "global_base_only"}:
            self.handles.append(
                self.model.ar_delta_head.register_forward_hook(self._zero_output)
            )
        if self.mode in {"no_fp_delta", "global_base_only"}:
            self.handles.append(
                self.model.fp_delta_head.register_forward_hook(self._zero_output)
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.no_grad()
def evaluate(model, loader, device, mode, amp, prediction_floor):
    model.eval()
    predictions, targets, drug_ids = [], [], []
    with Intervention(model, mode) as intervention:
        for batch in loader:
            intervention.set_batch(batch["drug_id"])
            batch = move_batch_to_device_caviar_subpocket(
                batch, device, non_blocking=True
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                details = model(batch, return_details=True)
            prediction = details["pred"].float()
            if prediction_floor > 0:
                prediction = prediction.clamp_min(prediction_floor)
            predictions.append(prediction.view(-1).cpu())
            targets.append(batch["label"].float().view(-1).cpu())
            drug_ids.extend(batch["drug_id"])
    prediction = torch.cat(predictions).numpy()
    target = torch.cat(targets).numpy()
    return metrics(target, prediction, drug_ids), target, prediction


def comparison(real_prediction, altered_prediction, real_mse, altered_mse):
    difference = altered_prediction - real_prediction
    return {
        "delta_mse_vs_real": float(altered_mse - real_mse),
        "max_abs_prediction_change": float(np.max(np.abs(difference))),
        "mean_abs_prediction_change": float(np.mean(np.abs(difference))),
        "rms_prediction_change": float(math.sqrt(np.mean(difference**2))),
        "prediction_correlation_with_real": float(
            np.corrcoef(real_prediction, altered_prediction)[0, 1]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    project = args.project.resolve()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    config = json.loads((run_dir / "run_config.json").read_text())
    split = json.loads(Path(config["split_json"]).read_text())
    checkpoint_path = run_dir / "checkpoint_best_val_mse.pt"
    device = torch.device(args.device)
    set_seed(int(config["seed"]))

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
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    loaders = {
        "val": DataLoader(Subset(dataset, split["val_indices"]), **loader_options),
        "test": DataLoader(Subset(dataset, split["test_indices"]), **loader_options),
    }

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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    results = {
        "audit": "CAVIAR_SP1_INDEPENDENT_CAUSAL_INTERVENTION_AUDIT",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split_json": config["split_json"],
        "seed": int(config["seed"]),
        "batch_size": args.batch_size,
        "prediction_floor": float(config["prediction_floor"]),
        "no_training_performed": True,
        "original_files_modified": False,
        "splits": {},
    }
    saved_arrays = {}
    for split_name, loader in loaders.items():
        split_results = {}
        predictions_by_mode = {}
        target_reference = None
        for mode in MODES:
            set_seed(int(config["seed"]))
            metric, target, prediction = evaluate(
                model,
                loader,
                device,
                mode,
                bool(config["amp"]),
                float(config["prediction_floor"]),
            )
            split_results[mode] = {"metrics": metric}
            predictions_by_mode[mode] = prediction
            target_reference = target
            print(
                f"{split_name.upper()} {mode:32s} "
                f"MSE={metric['mse']:.9f} CI={metric['ci']:.6f} "
                f"RM2={metric['rm2']:.6f}",
                flush=True,
            )
        real_mse = split_results["real"]["metrics"]["mse"]
        real_prediction = predictions_by_mode["real"]
        for mode in MODES:
            split_results[mode]["vs_real"] = comparison(
                real_prediction,
                predictions_by_mode[mode],
                real_mse,
                split_results[mode]["metrics"]["mse"],
            )
            saved_arrays[f"{split_name}_{mode}_prediction"] = predictions_by_mode[mode]
        saved_arrays[f"{split_name}_target"] = target_reference
        results["splits"][split_name] = split_results

    original_val = np.load(run_dir / "best_val_predictions.npz")
    original_test = np.load(
        run_dir / "final_test_predictions_minimum_validation_mse.npz"
    )
    results["provenance"] = {
        "val_target_max_abs_diff": float(
            np.max(np.abs(saved_arrays["val_target"] - original_val["target"]))
        ),
        "val_prediction_max_abs_diff": float(
            np.max(
                np.abs(saved_arrays["val_real_prediction"] - original_val["prediction"])
            )
        ),
        "test_target_max_abs_diff": float(
            np.max(np.abs(saved_arrays["test_target"] - original_test["target"]))
        ),
        "test_prediction_max_abs_diff": float(
            np.max(
                np.abs(
                    saved_arrays["test_real_prediction"]
                    - original_test["prediction"]
                )
            )
        ),
    }
    (output_dir / "audit_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "audit_predictions.npz", **saved_arrays)
    print("PROVENANCE", json.dumps(results["provenance"], sort_keys=True))
    print(f"OUTPUT_DIR={output_dir}")


if __name__ == "__main__":
    main()
