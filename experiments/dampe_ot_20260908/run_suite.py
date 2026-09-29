"""Resumable serial F0-F4 development suite; never evaluates test or launches folds 2-5."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import OUTPUT, dump, sha256


def run(a):
    root=OUTPUT/f'fold_{a.fold}'
    script=Path(__file__).resolve().parent
    lock=root/'suite.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    logs=root/'logs';logs.mkdir(exist_ok=True)
    started=time.time()
    status=dict(status='running',pid=os.getpid(),started_unix=started,fold=a.fold,
        modes=['F0','F1','F2','F3','F4'],seeds=[42,43,44],device=a.device,
        scope='development_validation_only_no_test_no_other_folds',completed=[])
    try:
        hashes={f.name:sha256(f) for f in script.glob('*.py')}
        dump(root/'suite_code_sha256.json',hashes)
        for seed in [42,43,44]:
            paired_init=None;paired_order=None
            for mode in ['F0','F1','F2','F3','F4']:
                name=f'{mode}_seed{seed}'
                status['current']=name;dump(root/'suite_status.json',status)
                out=root/'runs'/name
                if not (out/'summary.json').exists():
                    if out.exists():
                        raise RuntimeError(f'Incomplete run requires explicit inspection: {out}')
                    cmd=[sys.executable,'-B','-u',str(script/'train_frozen_ot.py'),
                         '--fold',str(a.fold),'--seed',str(seed),'--mode',mode,'--device',a.device]
                    print('START',name,flush=True)
                    with open(logs/(name+'.log'),'x') as log:
                        subprocess.run(cmd,cwd=str(script),stdout=log,stderr=subprocess.STDOUT,check=True)
                summary=json.loads((out/'summary.json').read_text())
                config=json.loads((out/'config.json').read_text())
                if not summary['complete'] or config['smoke']:
                    raise RuntimeError('Not a completed formal run')
                if paired_init is not None:
                    if paired_init!=config['initial_trainable_parameters_sha256']:
                        raise RuntimeError('Unpaired initialization')
                    if paired_order!=config['first_epoch_sampler_order_sha256']:
                        raise RuntimeError('Unpaired sampling')
                paired_init=config['initial_trainable_parameters_sha256']
                paired_order=config['first_epoch_sampler_order_sha256']
                status['completed'].append(summary)
                dump(root/'suite_status.json',status)
                print('DONE',name,summary['best_val_metrics'],flush=True)
        status['status']='complete';status['elapsed_seconds']=time.time()-started
        dump(root/'suite_status.json',status)
        subprocess.run([sys.executable,'-B',str(script/'summarize.py'),'--fold',str(a.fold)],check=True)
    except BaseException as exc:
        status['status']='failed';status['error']=repr(exc)
        dump(root/'suite_status.json',status)
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=1);p.add_argument('--device',default='cpu')
    run(p.parse_args())
