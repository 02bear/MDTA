import argparse
import json
import time
import numpy as np
import torch
from common import OUTPUT, dump, sha256, assert_sources
from ot_alignment import fit_alignment, compute_cost_matrix, sinkhorn_transport


def prepare(fold, epsilon, bootstrap_repeats):
    torch.set_num_threads(1)
    root = OUTPUT/f'fold_{fold}'
    cachefile = root/'cache/features.pt'
    cache = torch.load(cachefile, map_location='cpu', weights_only=False)
    assert_sources(cache)
    out = root/'ot'
    out.mkdir(exist_ok=False)
    manifests = {}
    started = time.monotonic()
    # Only drug OT is fitted for F0-F4. Protein fitting is a later experiment.
    e = cache['entities']['drug']
    ids = e['train_ids']
    rows = [e['ids'].index(i) for i in ids]
    x, y = e['h3'][rows], e['h1'][rows]
    for name, shuffle_seed in [('drug',None),('drug_shuffle',20260908)]:
        t, meta = fit_alignment(x,y,ids,ids,ids,epsilon,shuffle_seed)
        meta.update(fold=fold, cache_sha256=sha256(cachefile),
                    split_sha256=cache['split_sha256'], checkpoint_sha256=cache['checkpoint_sha256'])
        torch.save({'T':t,'metadata':meta},out/(name+'_ot_matrix.pt'))
        dump(out/(name+'_metadata.json'),meta)
        manifests[name]=meta
        print('OT_FIT',name,json.dumps({k:v for k,v in meta.items() if k not in ['diagnostics','train_entity_ids','target_permutation']}),flush=True)
    # Paired train-only bootstrap diagnoses plan and mapped-feature stability.
    rng=np.random.default_rng(20260909)
    ref=torch.load(out/'drug_ot_matrix.pt',weights_only=False)['T'].numpy()
    original=x.numpy().astype(np.float64)@(ref*ref.shape[1])
    trials=[]
    for k in range(bootstrap_repeats):
        ix=rng.integers(0,len(ids),len(ids))
        try:
            tb, info=sinkhorn_transport(compute_cost_matrix(x[ix],y[ix]),epsilon)
            z=x.numpy()@(tb*tb.shape[1])
            trials.append(dict(repeat=k,converged=True,iterations=info['iterations'],
                relative_plan_frobenius=float(np.linalg.norm(tb-ref)/max(np.linalg.norm(ref),1e-30)),
                relative_mapping_frobenius=float(np.linalg.norm(z-original)/max(np.linalg.norm(original),1e-30))))
        except RuntimeError as exc:
            trials.append(dict(repeat=k,converged=False,error=str(exc)))
        print('BOOTSTRAP',k+1,bootstrap_repeats,flush=True)
    dump(out/'bootstrap.json',dict(repeats=bootstrap_repeats,seed=20260909,
        fitting_population='unique_train_drugs_paired_resampling',trials=trials))
    dump(out/'manifest.json',dict(fold=fold,epsilon=epsilon,cache_sha256=sha256(cachefile),
        matrices={name:sha256(out/(name+'_ot_matrix.pt')) for name in manifests},
        elapsed_seconds=time.monotonic()-started,complete=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=1)
    p.add_argument('--epsilon',type=float,default=1e-3)
    p.add_argument('--bootstrap-repeats',type=int,default=20)
    a=p.parse_args();prepare(a.fold,a.epsilon,a.bootstrap_repeats)
