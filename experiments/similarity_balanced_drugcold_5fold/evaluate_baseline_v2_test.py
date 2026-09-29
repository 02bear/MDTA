"""Evaluate frozen inner-selected baseline checkpoints on disjoint outer tests."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

PROJECT = Path('/data1/ztx/MyModel-MDTA')
sys.path.insert(0, str(PROJECT))
import train_p13d_earlystop as t

ROOT = PROJECT / 'outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final'
OUT = ROOT / 'baseline_frozen_checkpoint_test_20260908'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

@torch.inference_mode()
def predict(model, loader, tag):
    model.eval()
    ps, ys = [], []
    for step, batch in enumerate(loader):
        batch = t.move_batch_to_device(batch, torch.device('cuda'))
        ps.append(model(batch).flatten().cpu())
        ys.append(batch['label'].flatten().cpu())
        if step % 50 == 0:
            print(f'{tag} batch={step}/{len(loader)}', flush=True)
    return torch.cat(ps), torch.cat(ys)

def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    rows, seen, all_p, all_y = [], [], [], []
    for f in range(1,6):
        folder = ROOT / f'baseline/fold_{f}'
        path = folder / 'best_model.pt'
        before = digest(path)
        ck = torch.load(path, map_location='cpu', weights_only=False)
        cfg = SimpleNamespace(**ck['args'])
        t.set_seed(cfg.seed)
        ds, train, val, _, vl = t.build_dataloaders(cfg)
        raw = pd.read_csv(cfg.pairs_csv, dtype={'drug_id':str,'protein_id':str})
        assert raw[['drug_id','protein_id']].equals(ds.df[['drug_id','protein_id']])
        assert np.allclose(raw.label, ds.df.label)
        split = json.loads(Path(cfg.split_json).read_text())
        saved = json.loads((folder/'split_indices.json').read_text())
        assert saved['train_indices']==train.indices and saved['val_indices']==val.indices
        ti = split['test_indices']
        assert len(ti)==len(set(ti))
        assert set(ti).isdisjoint(train.indices) and set(ti).isdisjoint(val.indices)
        trids=set(ds.df.iloc[train.indices].drug_id)
        vids=set(ds.df.iloc[val.indices].drug_id)
        frame=ds.df.iloc[ti].copy()
        ids=set(frame.drug_id)
        assert ids==set(map(str,split['outer_test_drugs']))
        assert ids.isdisjoint(trids|vids)
        assert not set(seen)&set(ti)
        model=t.build_model(cfg,torch.device('cuda'))
        model.load_state_dict(ck['model_state_dict'],strict=True)
        vp,vy=predict(model,vl,f'fold{f}_val_reproduction')
        vm=t.compute_regression_metrics(vp,vy)
        delta={k:abs(vm[k]-ck['val_metrics'][k]) for k in vm}
        assert max(delta.values())<1e-4,delta
        loader=DataLoader(Subset(ds,ti),batch_size=cfg.batch_size,shuffle=False,num_workers=0,collate_fn=t.mdta_collate_fn_p13d)
        pred,y=predict(model,loader,f'fold{f}_TEST')
        assert np.allclose(frame.label,y.numpy())
        metrics=t.compute_regression_metrics(pred,y)
        frame['pred']=pred.numpy()
        drugrows=[]
        for drug,g in frame.groupby('drug_id',sort=True):
            dy=torch.tensor(g.label.to_numpy(),dtype=torch.float32)
            dp=torch.tensor(g.pred.to_numpy(),dtype=torch.float32)
            drugrows.append({'drug_id':drug,'n':len(g),**t.compute_regression_metrics(dp,dy)})
        result={'fold':f,'best_epoch':ck['epoch'],'train_drugs':len(trids),'validation_drugs':len(vids),'test_drugs':len(ids),'test_pairs':len(ti),'validation_metrics':vm,'validation_reproduction_abs_diff':delta,'test_metrics':metrics,'per_drug':drugrows,'checkpoint_sha256':before,'split_sha256':digest(Path(cfg.split_json)),'refit_performed':False,'test_used_for_training_or_epoch_selection':False}
        assert before==digest(path)
        np.savez_compressed(OUT/f'fold_{f}_test_predictions.npz',indices=np.array(ti),drug_ids=frame.drug_id.to_numpy(),protein_ids=frame.protein_id.to_numpy(),y_true=y.numpy(),y_pred=pred.numpy())
        (OUT/f'fold_{f}_test_metrics.json').write_text(json.dumps(result,indent=2))
        rows.append(result);seen.extend(ti);all_p.append(pred);all_y.append(y)
        print('FOLD_RESULT='+json.dumps({'fold':f,'metrics':metrics}),flush=True)
        del model,ck,ds,loader,vl
        torch.cuda.empty_cache()
    assert sorted(seen)==list(range(len(raw)))
    macro={k:{'mean':float(np.mean([r['test_metrics'][k] for r in rows])),'sample_std':float(np.std([r['test_metrics'][k] for r in rows],ddof=1))} for k in rows[0]['test_metrics']}
    pooled=t.compute_regression_metrics(torch.cat(all_p),torch.cat(all_y))
    summary={'protocol':'frozen inner-validation-selected checkpoint evaluated on outer test; no outer refit','labels_used_for_split_stratification':True,'all_68_drugs_tested_once':True,'all_30056_pairs_tested_once':True,'folds':rows,'macro':macro,'pooled':pooled}
    (OUT/'TEST_SUMMARY.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE='+json.dumps({'macro':macro,'pooled':pooled}),flush=True)

if __name__=='__main__':
    main()
