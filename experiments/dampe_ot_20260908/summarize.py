import argparse
import json
import numpy as np
from common import OUTPUT,dump

def summarize(fold):
    root=OUTPUT/f'fold_{fold}';runs={};table=[];paired={}
    for mode in ['F0','F1','F2','F3','F4']:
        entries=[]
        for seed in [42,43,44]:
            p=root/'runs'/f'{mode}_seed{seed}'/'summary.json'
            if p.exists():
                value=json.loads(p.read_text());entries.append(value);runs[(mode,seed)]=value
        if entries:
            table.append(dict(mode=mode,completed_seeds=[e['seed'] for e in entries],
                validation={k:dict(mean=float(np.mean([e['best_val_metrics'][k] for e in entries])),
                sample_std=float(np.std([e['best_val_metrics'][k] for e in entries],ddof=1)) if len(entries)>1 else None)
                for k in ['mse','ci','rm2']}))
    for mode in ['F1','F2','F3','F4']:
        paired[mode]=[]
        for seed in [42,43,44]:
            if (mode,seed) in runs and ('F0',seed) in runs:
                paired[mode].append(dict(seed=seed,mse_change=runs[(mode,seed)]['best_val_metrics']['mse']-runs[('F0',seed)]['best_val_metrics']['mse']))
    output=dict(fold=fold,protocol='development_frozen_encoder',table=table,
        paired_delta_vs_F0=paired,complete=len(runs)==15,
        caveat='Validation selected both original encoder checkpoint and downstream checkpoint. Development evidence only. Seeds vary downstream training, not frozen encoder. Test not evaluated.')
    dump(root/'summary.json',output)
    print(json.dumps(output,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=1)
    summarize(p.parse_args().fold)
