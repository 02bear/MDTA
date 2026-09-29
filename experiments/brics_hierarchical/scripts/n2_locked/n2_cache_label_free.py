#!/usr/bin/env python3
"""Cache fold-specific frozen P13D pair features without affinity labels."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


def absolute(project, value):
    path = Path(value)
    return path if path.is_absolute() else project / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
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
        drug_2d_dir=absolute(args.project, saved["drug_2d_dir"]), use_drug_2d=False,
        drug_3d_dir=absolute(args.project, saved["drug_3d_dir"]), use_drug_3d=True,
    )
    model = MyModelMDTAP13D(
        drug_1d_in_dim=saved["drug_1d_in_dim"], drug_3d_node_in_dim=saved["drug_3d_node_in_dim"],
        protein_1d_in_dim=1280, protein_3d_node_s_dim=6, protein_3d_node_v_dim=3,
        hidden_dim=saved["hidden_dim"], dropout=saved["dropout"], task="regression",
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    first_drug, first_protein = {}, {}
    for idx, row in dataset.df.iterrows():
        first_drug.setdefault(str(row["drug_id"]), idx)
        first_protein.setdefault(str(row["protein_id"]), idx)

    def encode_unique(index_by_id, side):
        ids, indices = list(index_by_id), list(index_by_id.values())
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=mdta_collate_fn_p13d, pin_memory=True)
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
            result.update({key: value.detach().cpu() for key, value in zip(ids[cursor:cursor + len(fused)], fused)})
            cursor += len(fused)
        assert cursor == len(ids)
        return result

    drug_features = encode_unique(first_drug, "drug")
    protein_features = encode_unique(first_protein, "protein")
    predictions, pair_features = [], []
    for start in range(0, len(dataset.df), 4096):
        frame = dataset.df.iloc[start:start + 4096]
        drug = torch.stack([drug_features[str(x)] for x in frame["drug_id"]]).to(args.device)
        protein = torch.stack([protein_features[str(x)] for x in frame["protein_id"]]).to(args.device)
        pair = torch.cat([drug, protein], dim=-1)
        pair_features.append(pair.cpu())
        predictions.append(model.decoder(pair).cpu().reshape(-1))
    predictions, pair_features = torch.cat(predictions), torch.cat(pair_features)

    audit_indices = list(range(min(8, len(dataset))))
    audit_batch = next(iter(DataLoader(Subset(dataset, audit_indices), batch_size=len(audit_indices),
                                       shuffle=False, num_workers=0, collate_fn=mdta_collate_fn_p13d)))
    direct = model(move_batch_to_device(audit_batch, args.device)).cpu().reshape(-1)
    max_abs = float((direct - predictions[audit_indices]).abs().max())
    if max_abs > 1e-5 or pair_features.shape != (len(dataset.df), 256):
        raise RuntimeError(f"cache equivalence failed: max_abs={max_abs}, shape={tuple(pair_features.shape)}")

    payload = {
        "prediction": predictions, "pair_feature": pair_features,
        "drug_id": [str(x) for x in dataset.df["drug_id"]],
        "protein_id": [str(x) for x in dataset.df["protein_id"]],
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "label_independent": True, "contains_affinity_label": False,
    }
    assert "label" not in payload
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    audit = {
        "rows": len(predictions), "pair_feature_shape": list(pair_features.shape),
        "finite_predictions": int(torch.isfinite(predictions).sum()),
        "direct_factorized_max_abs_difference": max_abs,
        "label_independent": True, "contains_affinity_label": False,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": payload["checkpoint_sha256"],
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
