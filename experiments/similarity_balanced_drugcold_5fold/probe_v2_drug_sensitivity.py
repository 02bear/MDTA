"""Inspect feature diversity and prediction interactions without training."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0, '/data1/ztx/MyModel-MDTA')
import train_p13d_earlystop as t

root = Path('/data1/ztx/MyModel-MDTA/outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final')
out = root / 'diagnosis_fold13_20260907'
torch.set_num_threads(4)
results = []
for fold in (1, 3):
    z = np.load(out / f'fold_{fold}_val_predictions.npz', allow_pickle=True)
    drugs = sorted(set(z['drug_ids']))
    proteins = sorted(set(z['protein_ids']))
    def matrix(key):
        lookup = {(d, p): x for d, p, x in zip(z['drug_ids'], z['protein_ids'], z[key])}
        return np.array([[lookup[d, p] for p in proteins] for d in drugs], dtype=float)
    pred, y = matrix('y_pred'), matrix('y_true')
    def describe(m):
        centered = m - m.mean(axis=1, keepdims=True)
        interaction = m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + m.mean()
        return {'drug_mean_range': float(np.ptp(m.mean(axis=1))),
                'interaction_rms': float(np.sqrt(np.mean(interaction ** 2))),
                'interaction_fraction_of_variance': float(np.mean(interaction ** 2) / np.var(m)),
                'max_centered_profile_difference': float(np.ptp(centered, axis=0).max()),
                'minimum_profile_correlation': float(np.corrcoef(m)[np.triu_indices(len(m), 1)].min())}
    ck = torch.load(root / f'baseline/fold_{fold}/best_model.pt', map_location='cpu', weights_only=False)
    cfg = SimpleNamespace(**ck['args'])
    ds, tr, val, _, _ = t.build_dataloaders(cfg)
    model = t.build_model(cfg, torch.device('cpu'))
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model.eval()
    allowed = ds.df.iloc[tr.indices + val.indices]
    ix = allowed.groupby('drug_id', sort=True).head(1).index.tolist()
    captures = {k: [] for k in ('drug_1d_encoder', 'drug_3d_encoder', 'drug_fusion')}
    handles = [getattr(model, k).register_forward_hook(lambda m, a, o, key=k: captures[key].append(o.detach().cpu().numpy())) for k in captures]
    raw1d = []
    with torch.inference_mode():
        for i in range(0, len(ix), 8):
            batch = t.mdta_collate_fn_p13d([ds[j] for j in ix[i:i+8]])
            raw1d.append(batch['drug_1d'].numpy())
            d1 = model.drug_1d_encoder(batch['drug_1d'])
            d3 = model.drug_3d_encoder(batch['drug_3d'])
            model.drug_fusion([d1, d3])
    feature_report = {}
    allids = ds.df.iloc[ix].drug_id.to_numpy()
    for name, values in {**captures, 'raw_drug_1d': raw1d}.items():
        a = np.concatenate(values).astype(float)
        feature_report[name] = {}
        for role, mask in [('train', ~np.isin(allids, drugs)), ('val', np.isin(allids, drugs))]:
            x = a[mask]
            feature_report[name][role] = {'n': len(x), 'mean_feature_std': float(x.std(axis=0).mean()), 'max_feature_range': float(np.ptp(x, axis=0).max()), 'mean_norm': float(np.linalg.norm(x, axis=1).mean())}
    r = {'fold': fold, 'prediction_structure': describe(pred), 'label_structure': describe(y), 'features': feature_report}
    results.append(r)
    print(json.dumps(r), flush=True)
(out / 'sensitivity.json').write_text(json.dumps(results, indent=2))
