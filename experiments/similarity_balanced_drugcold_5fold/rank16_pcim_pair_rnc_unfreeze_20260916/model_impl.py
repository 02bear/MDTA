"""Frozen protocol: P13D encoders + additive/low-rank bilinear head, rank 16."""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

PROJECT = Path('/data1/ztx/MyModel-MDTA')
sys.path.insert(0, str(Path(__file__).resolve().parent / 'source'))
import train_p13d_earlystop as base

RANK = 16

class AdditiveBilinearHead(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.drug_main = nn.Linear(hidden_dim, 1)
        self.protein_main = nn.Linear(hidden_dim, 1, bias=False)
        self.drug_projection = nn.Linear(hidden_dim, RANK, bias=False)
        self.protein_projection = nn.Linear(hidden_dim, RANK, bias=False)

    def forward(self, pair):
        drug, protein = self.dropout(pair).split(self.hidden_dim, dim=-1)
        interaction = (self.drug_projection(drug) * self.protein_projection(protein)).sum(-1, keepdim=True) / math.sqrt(RANK)
        return self.drug_main(drug) + self.protein_main(protein) + interaction

_build = base.build_model
_save = base.save_checkpoint
_last_validation = {}

def build_model(args, device):
    model = _build(args, device)
    model.decoder = AdditiveBilinearHead(args.hidden_dim, args.dropout).to(device)
    args.head_type = 'additive_bilinear_rank16'
    args.head_rank = RANK
    args.initialization = 'fresh_seed42_no_checkpoint_weights'
    return model

def interaction_stats(pred, target, ids, proteins):
    def matrix(values):
        import pandas as pd
        return pd.DataFrame({'d': ids, 'p': proteins, 'v': values}).pivot(index='d', columns='p', values='v').to_numpy(dtype=float)
    pm, ym = matrix(pred), matrix(target)
    assert np.isfinite(pm).all() and np.isfinite(ym).all()
    stats = {}
    for name, a in [('pred', pm), ('label', ym)]:
        x = a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()
        stats[name + '_interaction_rms'] = float(np.sqrt(np.mean(x*x)))
    stats['per_drug_mse'] = np.mean((pm-ym)**2, axis=1).tolist()
    mask = target >= 7
    stats['label_ge7_mse'] = float(np.mean((pred[mask]-target[mask])**2)) if mask.any() else None
    return stats

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    preds, targets = [], []
    total = 0.0
    for batch in loader:
        batch = base.move_batch_to_device(batch, device)
        pred, target = model(batch), batch['label']
        total += float(criterion(pred, target).item()) * target.size(0)
        preds.append(pred.detach().cpu())
        targets.append(target.detach().cpu())
    pred, target = torch.cat(preds), torch.cat(targets)
    metrics = base.compute_regression_metrics(pred, target)
    metrics['loss'] = total / len(loader.dataset)
    frame = loader.dataset.dataset.df.iloc[loader.dataset.indices]
    assert np.allclose(frame.label, target.numpy().reshape(-1))
    _last_validation.clear()
    _last_validation.update(y_pred=pred.numpy().reshape(-1), y_true=target.numpy().reshape(-1),
                            drug_ids=frame.drug_id.to_numpy(), protein_ids=frame.protein_id.to_numpy(),
                            indices=np.asarray(loader.dataset.indices))
    metrics.update(interaction_stats(_last_validation['y_pred'], _last_validation['y_true'], frame.drug_id.to_numpy(), frame.protein_id.to_numpy()))
    print('VALIDATION_DIAGNOSTICS=' + json.dumps(metrics), flush=True)
    return metrics

def save_checkpoint(path, model, optimizer, epoch, train_metrics, val_metrics, args):
    _save(path, model, optimizer, epoch, train_metrics, val_metrics, args)
    if Path(path).name == 'best_model.pt':
        np.savez_compressed(Path(path).parent / 'best_val_predictions.npz', **_last_validation)

def main():
    base.build_model = build_model
    base.evaluate = evaluate
    base.save_checkpoint = save_checkpoint
    print('PROTOCOL=bilinear_rank16_fold1_fold3_development; FRESH_START=True; TEST_EVALUATED=False', flush=True)
    base.main()

if __name__ == '__main__':
    main()
