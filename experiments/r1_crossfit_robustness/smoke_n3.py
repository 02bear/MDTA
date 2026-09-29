#!/usr/bin/env python3
"""Non-training smoke checks for N3 split, label isolation, and one P13D forward."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from n3_common import (
    BATCH_SIZE, SelectiveLabelP13DDataset, build_model, build_n3_split,
    collate_with_index, load_selected_labels, read_pair_entities, rows_for_drugs, set_seed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    sys.path.insert(0, str(args.project))
    entities = read_pair_entities(args.pairs_csv)
    for fold in range(1, 6):
        seen = []
        for cf in range(1, 6):
            split = build_n3_split(args.project, fold, cf)
            seen.extend(split["holdout_drugs"])
            assert not set(split["holdout_drugs"]) & set(split["T_j"])
            assert set(split["epoch_train_drugs"]) | set(split["epoch_val_drugs"]) == set(split["T_j"])
        assert len(seen) == len(set(seen))

    split = build_n3_split(args.project, 1, 1)
    et_rows = rows_for_drugs(entities, split["epoch_train_drugs"])
    ev_rows = rows_for_drugs(entities, split["epoch_val_drugs"])
    h_rows = rows_for_drugs(entities, split["holdout_drugs"])
    labels = load_selected_labels(args.pairs_csv, et_rows + ev_rows, len(entities))
    assert torch.isnan(labels[h_rows]).all()
    dataset = SelectiveLabelP13DDataset(args.project, entities, labels, cache_features=True)
    uncached_dataset = SelectiveLabelP13DDataset(args.project, entities, labels, cache_features=False)
    data_loader = DataLoader(Subset(dataset, et_rows[:BATCH_SIZE]), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, collate_fn=collate_with_index)
    uncached_loader = DataLoader(Subset(uncached_dataset, et_rows[:BATCH_SIZE]), batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0, collate_fn=collate_with_index)
    set_seed(42)
    model = build_model(args.device)
    model.eval()
    batch = next(iter(data_loader))
    uncached_batch = next(iter(uncached_loader))
    from datasets.collate_p13d import move_batch_to_device
    batch = move_batch_to_device(batch, args.device)
    uncached_batch = move_batch_to_device(uncached_batch, args.device)
    for key in ("drug_1d", "label", "index"):
        assert torch.equal(batch[key], uncached_batch[key]), key
    for group in ("drug_3d", "protein_3d"):
        assert set(batch[group]) == set(uncached_batch[group])
        for key in batch[group]:
            assert torch.equal(batch[group][key], uncached_batch[group][key]), f"{group}.{key}"
    assert torch.equal(batch["protein_1d"], uncached_batch["protein_1d"])
    with torch.no_grad():
        prediction = model(batch)
        uncached_prediction = model(uncached_batch)
    assert prediction.shape[0] == len(et_rows[:BATCH_SIZE])
    assert torch.isfinite(prediction).all()
    max_abs = float(torch.max(torch.abs(prediction-uncached_prediction)))
    assert max_abs < 1e-5, max_abs
    print("N3_SPLIT_SMOKE=True")
    print("N3_HOLDOUT_LABEL_ISOLATION_SMOKE=True")
    print("N3_P13D_FORWARD_SMOKE=True")
    print("N3_CACHED_UNCACHED_INPUTS_EXACT=True")
    print(f"N3_CACHED_UNCACHED_FORWARD_MAX_ABS={max_abs:.12g}")


if __name__ == "__main__":
    main()
