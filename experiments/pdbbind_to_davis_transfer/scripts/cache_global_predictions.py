#!/usr/bin/env python3
"""Cache frozen baseline predictions in original Davis row order."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


def absolute(project, value):
    path = Path(value)
    return path if path.is_absolute() else project / path


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, default=Path("/data1/ztx/MyModel-MDTA"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    sys.path.insert(0, str(args.project))

    from datasets.davis_dataset_p13d import DavisDatasetP13D
    from datasets.collate_p13d import mdta_collate_fn_p13d, move_batch_to_device
    from models.model_p13d import MyModelMDTAP13D

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    dataset = DavisDatasetP13D(
        pairs_csv=absolute(args.project, saved["pairs_csv"]),
        drug_1d_dir=absolute(args.project, saved["drug_1d_dir"]),
        protein_1d_dir=absolute(args.project, saved["protein_1d_dir"]),
        protein_3d_dir=absolute(args.project, saved["protein_3d_dir"]),
        drug_2d_dir=absolute(args.project, saved["drug_2d_dir"]),
        use_drug_2d=False,
        drug_3d_dir=absolute(args.project, saved["drug_3d_dir"]),
        use_drug_3d=True,
    )
    model = MyModelMDTAP13D(
        drug_1d_in_dim=saved["drug_1d_in_dim"],
        drug_3d_node_in_dim=saved["drug_3d_node_in_dim"],
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=saved["hidden_dim"],
        dropout=saved["dropout"],
        task="regression",
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    first_drug, first_protein = {}, {}
    for idx, row in dataset.df.iterrows():
        first_drug.setdefault(row["drug_id"], idx)
        first_protein.setdefault(row["protein_id"], idx)

    def encode_unique(index_by_id, side):
        ids, indices = list(index_by_id), list(index_by_id.values())
        loader = DataLoader(
            Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=mdta_collate_fn_p13d, pin_memory=True,
        )
        result, cursor = {}, 0
        for batch in loader:
            batch = move_batch_to_device(batch, args.device)
            if side == "drug":
                one = model.drug_1d_encoder(batch["drug_1d"])
                three = model.drug_3d_encoder(batch["drug_3d"])
                fused = model.drug_fusion([one, three])
            else:
                one = model.protein_1d_encoder(batch["protein_1d"])
                three = model.protein_3d_encoder(batch["protein_3d"])
                fused = model.protein_fusion([one, three])
            batch_ids = ids[cursor:cursor + fused.shape[0]]
            result.update({key: value for key, value in zip(batch_ids, fused.detach().cpu())})
            cursor += fused.shape[0]
        if cursor != len(ids):
            raise RuntimeError(f"{side} unique encoding count mismatch")
        return result

    drug_features = encode_unique(first_drug, "drug")
    protein_features = encode_unique(first_protein, "protein")
    predictions = []
    for start in range(0, len(dataset.df), 4096):
        frame = dataset.df.iloc[start:start + 4096]
        drug = torch.stack([drug_features[x] for x in frame["drug_id"]]).to(args.device)
        protein = torch.stack([protein_features[x] for x in frame["protein_id"]]).to(args.device)
        predictions.append(model.decoder(torch.cat([drug, protein], dim=-1)).cpu().reshape(-1))
    predictions = torch.cat(predictions)

    # Numerical proof that factorized caching is identical to ordinary forward.
    audit_indices = list(range(min(8, len(dataset))))
    audit_batch = next(iter(DataLoader(
        Subset(dataset, audit_indices), batch_size=len(audit_indices), shuffle=False,
        num_workers=0, collate_fn=mdta_collate_fn_p13d,
    )))
    direct = model(move_batch_to_device(audit_batch, args.device)).cpu().reshape(-1)
    factorized = predictions[audit_indices]
    equivalence_max_abs = float((direct - factorized).abs().max())
    if equivalence_max_abs > 1e-5:
        raise RuntimeError(f"factorized/global mismatch: {equivalence_max_abs}")
    if len(predictions) != len(dataset.df):
        raise RuntimeError("prediction count does not match dataset")
    payload = {
        "prediction": predictions,
        "label": torch.tensor(dataset.df["label"].to_numpy(), dtype=torch.float32),
        "drug_id": dataset.df["drug_id"].tolist(),
        "protein_id": dataset.df["protein_id"].tolist(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_metrics": checkpoint.get("val_metrics"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    audit = {
        "rows": len(predictions),
        "finite_predictions": int(torch.isfinite(predictions).sum()),
        "mean": float(predictions.mean()),
        "std": float(predictions.std()),
        "unique_drugs_encoded": len(drug_features),
        "unique_proteins_encoded": len(protein_features),
        "direct_factorized_max_abs_difference": equivalence_max_abs,
        "checkpoint": str(args.checkpoint),
        "test_metrics_computed": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
