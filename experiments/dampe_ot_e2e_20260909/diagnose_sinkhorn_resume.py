"""Recompute the next periodic OT maps from a saved checkpoint without training."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_periodic_ot import (OUTPUT_ROOT, SPLIT_ROOT, build_model, make_dataset,
                               unique_entity_loaders, update_ot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint_path = OUTPUT_ROOT/args.experiment/f'fold_{args.fold}'/'latest_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    dataset = make_dataset()
    split = json.loads((SPLIT_ROOT/f'fold_{args.fold}'/'split.json').read_text())
    entities = unique_entity_loaders(dataset, split, checkpoint['args']['batch_size'])
    previous = {
        'drug': model.drug_ot_map.detach().cpu().clone(),
        'protein': model.protein_ot_map.detach().cpu().clone(),
    }
    _, metadata = update_ot(model, entities, device, checkpoint['args']['epsilon'], previous)
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
