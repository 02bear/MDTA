#!/usr/bin/env python3
"""Shared, locked utilities for Experiment N3."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SEED = 42
N_PROTEINS = 442
MAX_EPOCHS = 500
PATIENCE = 60
MIN_DELTA = 1e-4
BATCH_SIZE = 16
LR = 3e-4
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 128
DROPOUT = 0.1

R1_CONFIGS = {
    1: {"gamma": 2.0, "k_drug": 8, "min_drug_similarity": 0.0, "tau": 0.05, "beta": 0.0, "clip": 1.0, "scale": 1.0},
    2: {"gamma": 1.0, "k_drug": 8, "min_drug_similarity": 0.2, "tau": 0.10, "beta": 0.0, "clip": 1.0, "scale": 1.0},
    3: {"gamma": 2.0, "k_drug": 8, "min_drug_similarity": 0.0, "tau": 0.05, "beta": 0.0, "clip": 1.0, "scale": 1.0},
    4: {"gamma": 2.0, "k_drug": 4, "min_drug_similarity": 0.0, "tau": 0.05, "beta": 0.5, "clip": 1.0, "scale": 0.75},
    5: {"gamma": 2.0, "k_drug": 4, "min_drug_similarity": 0.0, "tau": 0.05, "beta": 0.0, "clip": 1.0, "scale": 1.0},
}

REPRO_TARGETS = {
    1: {"P13D": 0.48946522331543174, "R1_in": 0.46135756446112697},
    2: {"P13D": 0.6772050433650999, "R1_in": 0.6136161010173478},
    3: {"P13D": 0.7725180673946208, "R1_in": 0.7399211297932111},
    4: {"P13D": 0.4743354582235450, "R1_in": 0.45667178684049803},
    5: {"P13D": 0.7334100130850374, "R1_in": 0.6692296481222448},
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def read_pair_entities(pairs_csv: Path) -> list[tuple[str, str]]:
    rows = []
    with pairs_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not {"drug_id", "protein_id", "label"}.issubset(fields):
            raise RuntimeError(f"unexpected pairs columns: {reader.fieldnames}")
        for row in reader:
            rows.append((str(row["drug_id"]), str(row["protein_id"])))
    return rows


def load_selected_labels(pairs_csv: Path, allowed_rows, total_rows: int) -> torch.Tensor:
    """Read label values only for explicitly allowed row indices."""
    allowed = {int(x) for x in allowed_rows}
    labels = torch.full((total_rows,), float("nan"), dtype=torch.float32)
    seen = set()
    with pairs_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "label" not in (reader.fieldnames or []):
            raise RuntimeError("pairs CSV has no label column")
        for index, row in enumerate(reader):
            if index in allowed:
                labels[index] = float(row["label"])
                seen.add(index)
    if seen != allowed:
        raise RuntimeError(f"selected labels missing: {sorted(allowed-seen)[:10]}")
    if allowed and not torch.isfinite(labels[list(allowed)]).all():
        raise RuntimeError("selected labels contain NaN/Inf")
    return labels


def rows_for_drugs(entities, drugs) -> list[int]:
    wanted = {str(x) for x in drugs}
    rows = [i for i, (drug, _) in enumerate(entities) if drug in wanted]
    observed = {entities[i][0] for i in rows}
    if observed != wanted:
        raise RuntimeError(f"drug mapping mismatch; missing={sorted(wanted-observed)}")
    counts = {drug: 0 for drug in wanted}
    for row in rows:
        counts[entities[row][0]] += 1
    bad = {drug: count for drug, count in counts.items() if count != N_PROTEINS}
    if bad:
        raise RuntimeError(f"expected {N_PROTEINS} pairs per drug: {bad}")
    return rows


def epoch_hash(outer_fold: int, cf_fold: int, drug_id: str) -> str:
    value = f"N3_EPOCH_V1|outer={outer_fold}|cf={cf_fold}|drug={drug_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_n3_split(project: Path, fold: int, cf_fold: int) -> dict:
    split_path = project / f"data/splits/davis_drug_cold_5fold_seed42/fold_{fold}/split.json"
    similarity_path = project / "experiments/klifs85_interaction/data/similarity_audit_fold1/entity_similarities.npz"
    outer = json.loads(split_path.read_text(encoding="utf-8"))
    similarity = np.load(similarity_path, allow_pickle=True)
    drug_ids = [str(x) for x in similarity["drug_ids"].tolist()]
    lookup = {drug: i for i, drug in enumerate(drug_ids)}
    outer_train = [str(x) for x in outer["train_drugs"]]
    train_indices = np.asarray([lookup[x] for x in outer_train], dtype=int)
    rng = np.random.default_rng(SEED)
    folds = [np.asarray(x, dtype=int) for x in np.array_split(rng.permutation(train_indices), 5)]
    if cf_fold not in range(1, 6):
        raise ValueError(cf_fold)
    holdout = [drug_ids[int(x)] for x in folds[cf_fold - 1]]
    holdout_set = set(holdout)
    training = [drug for drug in outer_train if drug not in holdout_set]
    ordered = sorted(training, key=lambda drug: epoch_hash(fold, cf_fold, drug))
    n_val = math.ceil(len(training) / 5)
    epoch_val = ordered[:n_val]
    epoch_train = ordered[n_val:]
    all_holdout = [drug_ids[int(x)] for values in folds for x in values]
    assert len(all_holdout) == len(set(all_holdout)) == len(outer_train)
    assert set(all_holdout) == set(outer_train)
    assert not set(epoch_train) & set(epoch_val)
    assert set(epoch_train) | set(epoch_val) == set(training)
    assert not holdout_set & set(training)
    return {
        "outer_fold": fold,
        "cf_fold": cf_fold,
        "seed": SEED,
        "outer_train_drugs": outer_train,
        "holdout_drugs": holdout,
        "T_j": training,
        "epoch_train_drugs": epoch_train,
        "epoch_val_drugs": epoch_val,
        "HOLDOUT_USED_FOR_TRAINING": False,
        "HOLDOUT_USED_FOR_EPOCH_SELECTION": False,
        "split_rule": "historical build_inner_folds: default_rng(42).permutation + array_split(5)",
        "epoch_rule": "SHA256 N3_EPOCH_V1; first ceil(len(T_j)/5) by hash is EpochVal",
    }


class SelectiveLabelP13DDataset(Dataset):
    """Historical P13D feature pipeline with an explicit label-access allowlist."""

    def __init__(self, project: Path, entities, labels: torch.Tensor, cache_features: bool = True):
        self.project = project
        self.entities = entities
        self.labels = labels
        self.cache_features = cache_features
        self._drug_cache = {}
        self._protein_cache = {}
        self.drug_1d = project / "data/processed/davis/drug_1d_chemberta2"
        self.drug_3d = project / "data/processed/davis/drug_3d"
        self.protein_1d = project / "data/processed/davis/protein_1d_esm2"
        self.protein_3d = project / "data/processed/davis/protein_3d_gvp"

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, index):
        drug_id, protein_id = self.entities[index]
        drug_pair = self._drug_cache.get(drug_id) if self.cache_features else None
        if drug_pair is None:
            drug_pair = (
                torch.load(self.drug_1d / f"{drug_id}.pt", weights_only=False),
                torch.load(self.drug_3d / f"{drug_id}.pt", weights_only=False),
            )
            if self.cache_features:
                self._drug_cache[drug_id] = drug_pair
        protein_pair = self._protein_cache.get(protein_id) if self.cache_features else None
        if protein_pair is None:
            protein_pair = (
                torch.load(self.protein_1d / f"{protein_id}.pt", weights_only=False),
                torch.load(self.protein_3d / f"{protein_id}.pt", weights_only=False),
            )
            if self.cache_features:
                self._protein_cache[protein_id] = protein_pair
        drug_1d_obj, drug_3d_obj = drug_pair
        protein_1d_obj, protein_3d_obj = protein_pair
        return {
            "index": int(index),
            "drug_id": drug_id,
            "protein_id": protein_id,
            "drug_1d": drug_1d_obj["mean"].float(),
            "drug_2d": None,
            "drug_3d": drug_3d_obj,
            "protein_1d": protein_1d_obj["mean"].float(),
            "protein_3d": {
                "node_s": protein_3d_obj["node_s"].float(),
                "node_v": protein_3d_obj["node_v"].float(),
                "coords": protein_3d_obj["coords"].float(),
                "edge_index": protein_3d_obj["edge_index"].long(),
                "edge_s": protein_3d_obj["edge_s"].float(),
                "edge_v": protein_3d_obj["edge_v"].float(),
            },
            "label": self.labels[index].view(1),
        }


def collate_with_index(batch):
    from datasets.collate_p13d import mdta_collate_fn_p13d
    indices = torch.tensor([sample.pop("index") for sample in batch], dtype=torch.long)
    result = mdta_collate_fn_p13d(batch)
    result["index"] = indices
    return result


def build_model(device):
    from models.model_p13d import MyModelMDTAP13D
    return MyModelMDTAP13D(
        drug_1d_in_dim=768,
        drug_3d_node_in_dim=10,
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        task="regression",
    ).to(device)


def locked_training_config() -> dict:
    return {
        "architecture": "models.model_p13d.MyModelMDTAP13D",
        "seed": SEED,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "optimizer": "torch.optim.Adam",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "loss": "MSELoss",
        "gradient_clip": None,
        "num_workers": 0,
    }
