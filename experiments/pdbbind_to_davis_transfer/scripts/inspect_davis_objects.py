#!/usr/bin/env python3
from pathlib import Path
import torch

root = Path('/data1/ztx/MyModel-MDTA/data/processed/davis')
for rel in [
    'drug_3d_richatom43/11314340.pt',
    'drug_atom_features_v2/11314340.pt',
    'protein_residue_features_v2/AAK1.pt',
    'protein_3d_gvp_residue_esm/AAK1.pt',
]:
    obj = torch.load(root / rel, map_location='cpu', weights_only=False)
    print('\nFILE', rel, 'TYPE', type(obj))
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(key, tuple(value.shape) if torch.is_tensor(value) else type(value).__name__)
    else:
        for key in getattr(obj, 'keys', lambda: [])():
            value = obj[key]
            print(key, tuple(value.shape) if torch.is_tensor(value) else type(value).__name__)
