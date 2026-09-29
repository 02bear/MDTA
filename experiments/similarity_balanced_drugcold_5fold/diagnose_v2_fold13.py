"""Post-hoc validation diagnosis. No training or test-set evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

PROJECT = Path('/data1/ztx/MyModel-MDTA')
sys.path.insert(0, str(PROJECT))
import train_p13d_earlystop as t

def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--similarity', required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cache = np.load(args.similarity, allow_pickle=True)
    ids = list(map(str, cache['drug_ids']))
    sim = cache['drug_similarity']
    pos = {d: i for i, d in enumerate(ids)}
    root = PROJECT / 'outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/baseline'
    all_results = []
    for fold in (1, 3):
        ckpath = root / f'fold_{fold}/best_model.pt'
        ck = torch.load(ckpath, map_location='cpu', weights_only=False)
        cfg = SimpleNamespace(**ck['args'])
        t.set_seed(cfg.seed)
        ds, train, val, _, loader = t.build_dataloaders(cfg)
        raw = pd.read_csv(cfg.pairs_csv, dtype={'drug_id': str, 'protein_id': str})
        assert raw[['drug_id', 'protein_id']].equals(ds.df[['drug_id', 'protein_id']])
        assert np.allclose(raw.label, ds.df.label)
        split = json.loads(Path(cfg.split_json).read_text())
        saved = json.loads((root / f'fold_{fold}/split_indices.json').read_text())
        assert saved['train_indices'] == train.indices and saved['val_indices'] == val.indices
        model = t.build_model(cfg, torch.device('cuda'))
        model.load_state_dict(ck['model_state_dict'], strict=True)
        model.eval()
        preds, targets = [], []
        with torch.inference_mode():
            for step, batch in enumerate(loader):
                batch = t.move_batch_to_device(batch, torch.device('cuda'))
                preds.append(model(batch).flatten().cpu())
                targets.append(batch['label'].flatten().cpu())
                if step % 50 == 0:
                    print(f'fold={fold} val_batch={step}/{len(loader)}', flush=True)
        pred, target = torch.cat(preds), torch.cat(targets)
        metrics = t.compute_regression_metrics(pred, target)
        delta = {k: abs(metrics[k] - ck['val_metrics'][k]) for k in metrics}
        assert max(delta.values()) < 1e-4, delta
        v = ds.df.iloc[val.indices].copy()
        assert np.allclose(v.label, target.numpy())
        v['pred'] = pred.numpy()
        tr = ds.df.iloc[train.indices]
        trids = sorted(tr.drug_id.unique())
        assert set(trids).isdisjoint(v.drug_id)
        matrix = tr.pivot(index='drug_id', columns='protein_id', values='label')
        train_mean = float(tr.label.mean())
        rows = []
        total_sse = float(((v.pred - v.label) ** 2).sum())
        for drug, g in v.groupby('drug_id'):
            y, yp = g.label.to_numpy(), g.pred.to_numpy()
            s = np.asarray([sim[pos[drug], pos[d]] for d in trids])
            order = np.argsort(-s, kind='stable')
            nnids = [trids[i] for i in order[:3]]
            profiles = matrix.loc[trids, g.protein_id].to_numpy()
            nn = profiles[order[0]]
            top3 = profiles[order[:3]].mean(axis=0)
            tm = profiles.mean(axis=0)
            err = yp - y
            per = {'drug_id': drug, 'n': len(g), 'mse': mse(y, yp),
                   'mae': float(np.abs(err).mean()), 'ci': t.get_cindex(y, yp),
                   'y_mean': float(y.mean()), 'pred_mean': float(yp.mean()),
                   'y_std': float(y.std()), 'pred_std': float(yp.std()),
                   'bias': float(err.mean()), 'bias_squared': float(err.mean() ** 2),
                   'centered_error_mse': float(err.var()),
                   'sse_share': float((err ** 2).sum() / total_sse),
                   'label5_fraction': float(np.isclose(y, 5).mean()),
                   'active_ge7_count': int((y >= 7).sum()),
                   'active_ge7_mse': mse(y[y >= 7], yp[y >= 7]) if (y >= 7).any() else None,
                   'active_ge7_sse_share': float((err[y >= 7] ** 2).sum() / (err ** 2).sum()),
                   'floor5_mse': mse(y[np.isclose(y, 5)], yp[np.isclose(y, 5)]) if np.isclose(y, 5).any() else None,
                   'top1_similarity': float(s[order[0]]),
                   'top3_similarity_mean': float(s[order[:3]].mean()),
                   'train_neighbors_ge04': int((s >= .4).sum()),
                   'nearest_ids': nnids, 'nearest_similarities': s[order[:3]].tolist(),
                   'nearest_profile_mse': mse(y, nn), 'top3_profile_mse': mse(y, top3),
                   'target_mean_mse': mse(y, tm), 'constant_train_mean_mse': mse(y, train_mean),
                   'nearest_profile_ci': t.get_cindex(y, nn),
                   'nearest_centered_profile_mse': mse(y-y.mean(), nn-nn.mean()),
                   'largest_errors': [{'protein_id': str(g.iloc[i].protein_id), 'y': float(y[i]), 'pred': float(yp[i])} for i in np.argsort(-(err ** 2))[:10]]}
            rows.append(per)
        rows.sort(key=lambda r: -r['mse'])
        old = json.loads((PROJECT / f'outputs/Refine_experiment/davis/cold_start/drug_cold_5fold_seed42/baseline/fold_{fold}/split_indices.json').read_text())
        oldids = set(ds.df.iloc[old['val_indices']].drug_id)
        result = {'fold': fold, 'epoch': ck['epoch'], 'metrics': metrics, 'reproduction_abs_diff': delta,
                  'checkpoint_sha256': hashlib.sha256(ckpath.read_bytes()).hexdigest(),
                  'train_drugs': len(trids), 'val_drugs': len(rows), 'old_same_fold_overlap': sorted(oldids & set(v.drug_id)),
                  'top2_sse_share': sum(r['sse_share'] for r in rows[:2]),
                  'remaining5_mse': float(np.mean([r['mse'] for r in rows[2:]])), 'drugs': rows}
        np.savez_compressed(out / f'fold_{fold}_val_predictions.npz', drug_ids=v.drug_id.to_numpy(), protein_ids=v.protein_id.to_numpy(), y_true=target.numpy(), y_pred=pred.numpy(), indices=np.asarray(val.indices))
        (out / f'fold_{fold}_diagnosis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
        all_results.append(result)
        del model, ck, ds, loader
        torch.cuda.empty_cache()
    (out / 'diagnosis.json').write_text(json.dumps(all_results, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
