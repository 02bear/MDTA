#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import torch


ROOT = Path("/data1/ztx/MyModel-MDTA")


def describe(value):
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return {"type": type(value).__name__, "preview": str(value)[:200]}


global_data = torch.load(
    ROOT / "experiments/pdbbind_to_davis_transfer/data/global_predictions/fold1_with_features.pt",
    map_location="cpu",
    weights_only=False,
)
print("GLOBAL", {key: describe(value) for key, value in global_data.items()})

klifs = torch.load(
    ROOT / "experiments/klifs85_interaction/data/klifs85_features.pt",
    map_location="cpu",
    weights_only=False,
)
print("KLIFS_TOP", klifs.keys())
first_protein = next(iter(klifs["proteins"]))
print("KLIFS_SAMPLE", first_protein, {key: describe(value) for key, value in klifs["proteins"][first_protein].items()})

pairs = pd.read_csv(ROOT / "data/raw/davis/pairs.csv", dtype={"drug_id": str, "protein_id": str})
first_drug = pairs.iloc[0]["drug_id"]
drug = torch.load(
    ROOT / "data/processed/davis/drug_3d" / f"{first_drug}.pt",
    map_location="cpu",
    weights_only=False,
)
print("DRUG_SAMPLE", first_drug, {key: describe(value) for key, value in drug.items()})

protein = torch.load(
    ROOT / "data/processed/davis/protein_3d_gvp" / f"{first_protein}.pt",
    map_location="cpu",
    weights_only=False,
)
print("PROTEIN_SAMPLE", first_protein, {key: describe(value) for key, value in protein.items()})
