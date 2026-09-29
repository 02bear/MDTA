"""Independent, selection-free replay of baseline and experiment-A Fold1 test metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from train_periodic_ot import (BASE_ROOT, OUTPUT_ROOT, SPLIT_ROOT, build_model,
                               make_dataset, validate_split)
from datasets.collate_p13d import mdta_collate_fn_p13d
from models.model_p13d import MyModelMDTAP13D
from train_p13d_earlystop import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    fold = 1
    dataset = make_dataset()
    split_path = SPLIT_ROOT/'fold_1'/'split.json'
    split = json.loads(split_path.read_text())
    split_audit = validate_split(dataset, split)
    test_indices = split['test_indices']
    loader = DataLoader(Subset(dataset, test_indices), batch_size=16, shuffle=False,
                        num_workers=0, collate_fn=mdta_collate_fn_p13d)
    criterion = torch.nn.MSELoss()

    baseline_checkpoint = torch.load(BASE_ROOT/'fold_1'/'best_model.pt',
                                     map_location=device, weights_only=False)
    baseline = MyModelMDTAP13D(hidden_dim=128, dropout=0.1, task='regression').to(device)
    baseline.load_state_dict(baseline_checkpoint['model_state_dict'], strict=True)
    baseline_metrics = evaluate(baseline, loader, criterion, device)
    del baseline

    a_path = OUTPUT_ROOT/'ot_warmstart_finetune_both'/'fold_1'/'best_model.pt'
    a_checkpoint = torch.load(a_path, map_location=device, weights_only=False)
    model_a = build_model(device)
    model_a.load_state_dict(a_checkpoint['model_state_dict'], strict=True)
    a_metrics = evaluate(model_a, loader, criterion, device)

    labels = dataset.df.iloc[test_indices]['label'].to_numpy(dtype=float)
    train_mean = float(dataset.df.iloc[split['train_indices']]['label'].mean())
    result = {
        'verification_only_not_used_for_selection': True,
        'split_audit': split_audit,
        'test_pairs': len(test_indices),
        'test_label_mean': float(labels.mean()),
        'test_label_std': float(labels.std()),
        'train_mean_constant_test_mse': float(np.mean((labels-train_mean)**2)),
        'baseline_best_epoch': baseline_checkpoint['epoch'],
        'baseline_test': baseline_metrics,
        'experiment_a_best_epoch': a_checkpoint['epoch'],
        'experiment_a_test': a_metrics,
        'experiment_a_stored_test_mse': json.loads(
            (OUTPUT_ROOT/'ot_warmstart_finetune_both'/'fold_1'/'metrics.json').read_text()
        )['test_mse'],
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
