#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


PROJECT = Path("/data1/ztx/MyModel-MDTA")
sys.path.insert(0, str(PROJECT))

from datasets.collate_p13d import mdta_collate_fn_p13d, move_batch_to_device
from datasets.davis_dataset_p13d import DavisDatasetP13D
from models.model_p13d_finegrained_residual import MyModelMDTAP13DFineGrained


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def mse(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


@torch.no_grad()
def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    checkpoint_path = PROJECT / "outputs/Refine_experiment/davis/cold_start/drug_cold_5fold_seed42/e2/fold_5/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]

    def absolute(value):
        path = Path(value)
        return path if path.is_absolute() else PROJECT / path

    dataset = DavisDatasetP13D(
        pairs_csv=absolute(args["pairs_csv"]),
        drug_1d_dir=absolute(args["drug_1d_dir"]),
        protein_1d_dir=absolute(args["protein_1d_dir"]),
        protein_3d_dir=absolute(args["protein_3d_dir"]),
        drug_2d_dir=absolute(args["drug_2d_dir"]),
        use_drug_2d=False,
        drug_3d_dir=absolute(args["drug_3d_dir"]),
        use_drug_3d=True,
    )
    split = json.loads(absolute(args["split_json"]).read_text())
    val_indices = split["val_indices"]
    loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=mdta_collate_fn_p13d,
        pin_memory=True,
    )
    model = MyModelMDTAP13DFineGrained(
        drug_1d_in_dim=args["drug_1d_in_dim"],
        drug_3d_node_in_dim=args["drug_3d_node_in_dim"],
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=args["hidden_dim"],
        dropout=args["dropout"],
        task="regression",
        pocket_top_k=args["pocket_top_k"],
        interaction_heads=args["interaction_heads"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    values = {"e2": [], "e2_base": [], "local_delta": [], "label": []}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        output = model(batch, return_debug=True)
        values["e2"].append(output["pred"].detach().cpu().reshape(-1))
        values["e2_base"].append(output["base_pred"].detach().cpu().reshape(-1))
        values["local_delta"].append(output["local_delta"].detach().cpu().reshape(-1))
        values["label"].append(batch["label"].detach().cpu().reshape(-1))
    values = {key: torch.cat(parts).numpy().astype(np.float64) for key, parts in values.items()}

    frame = dataset.df.iloc[val_indices][["drug_id", "protein_id", "label"]].copy().reset_index(drop=True)
    frame["drug_id"] = frame["drug_id"].astype(str)
    frame["protein_id"] = frame["protein_id"].astype(str)
    frame["e2"] = values["e2"]
    frame["e2_base"] = values["e2_base"]
    frame["local_delta"] = values["local_delta"]

    kernel_path = PROJECT / "experiments/klifs85_interaction/outputs/residual_kernel_fold5_v2/outer_validation_predictions.csv"
    kernel = pd.read_csv(kernel_path, dtype={"drug_id": str, "protein_id": str})
    keep = ["drug_id", "protein_id", "R0", "R1", "R1_correction", "R2_active"]
    frame = frame.merge(kernel[keep], on=["drug_id", "protein_id"], how="inner", validate="one_to_one")
    if len(frame) != len(val_indices):
        raise RuntimeError(f"alignment mismatch: {len(frame)} versus {len(val_indices)}")
    if np.max(np.abs(frame["label"].to_numpy() - values["label"])) > 1e-5:
        raise RuntimeError("label order mismatch")

    sim = np.load(PROJECT / "experiments/klifs85_interaction/data/similarity_audit_fold1/entity_similarities.npz")
    drug_ids = [str(x) for x in sim["drug_ids"].tolist()]
    lookup = {drug: i for i, drug in enumerate(drug_ids)}
    train_drugs = [str(x) for x in split["train_drugs"]]
    train_idx = np.asarray([lookup[x] for x in train_drugs], dtype=int)
    drug_similarity = sim["drug_similarity"].astype(np.float64)

    per_drug = []
    for drug_id, group in frame.groupby("drug_id", sort=False):
        similarities = np.sort(drug_similarity[lookup[drug_id], train_idx])[::-1]
        y = group["label"].to_numpy()
        row = {
            "drug_id": drug_id,
            "n_pairs": len(group),
            "top1_train_similarity": float(similarities[0]),
            "top8_train_similarity_mean": float(similarities[:8].mean()),
            "r0_mse": mse(y, group["R0"]),
            "r1_mse": mse(y, group["R1"]),
            "e2_base_mse": mse(y, group["e2_base"]),
            "e2_mse": mse(y, group["e2"]),
            "r1_gain": mse(y, group["R0"]) - mse(y, group["R1"]),
            "e2_gain": mse(y, group["R0"]) - mse(y, group["e2"]),
            "e2_local_gain_vs_e2_base": mse(y, group["e2_base"]) - mse(y, group["e2"]),
            "r1_correction_abs_mean": float(group["R1_correction"].abs().mean()),
            "e2_total_correction_abs_mean": float((group["e2"] - group["R0"]).abs().mean()),
            "e2_local_delta_abs_mean": float(group["local_delta"].abs().mean()),
        }
        per_drug.append(row)
    per_drug = pd.DataFrame(per_drug).sort_values("e2_gain", ascending=False)

    y = frame["label"].to_numpy()
    true_r0_residual = y - frame["R0"].to_numpy()
    true_e2_base_residual = y - frame["e2_base"].to_numpy()
    summary = {
        "guardrail": "validation indices only; test indices and metrics not accessed",
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_args": args,
        "n_validation_pairs": len(frame),
        "n_validation_drugs": int(frame["drug_id"].nunique()),
        "mse": {
            "frozen_p13d_R0": mse(y, frame["R0"]),
            "drug_residual_R1": mse(y, frame["R1"]),
            "E2_retrained_global_base": mse(y, frame["e2_base"]),
            "E2_total": mse(y, frame["e2"]),
        },
        "e2_local_increment_mse_gain": mse(y, frame["e2_base"]) - mse(y, frame["e2"]),
        "correlations": {
            "R1_correction_vs_true_R0_residual": corr(frame["R1_correction"], true_r0_residual),
            "E2_total_correction_vs_true_R0_residual": corr(frame["e2"] - frame["R0"], true_r0_residual),
            "E2_local_delta_vs_true_E2_base_residual": corr(frame["local_delta"], true_e2_base_residual),
            "top1_similarity_vs_R1_gain_by_drug": corr(per_drug["top1_train_similarity"], per_drug["r1_gain"]),
        },
        "mean_abs_correction": {
            "R1": float(frame["R1_correction"].abs().mean()),
            "E2_total_vs_R0": float((frame["e2"] - frame["R0"]).abs().mean()),
            "E2_local_delta": float(frame["local_delta"].abs().mean()),
        },
    }
    output_dir = PROJECT / "experiments/klifs85_interaction/outputs/fold5_e2_vs_kernel_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_drug.to_csv(output_dir / "per_drug.csv", index=False)
    frame.to_csv(output_dir / "per_pair.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(per_drug.to_string(index=False))


if __name__ == "__main__":
    main()
