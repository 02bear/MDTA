"""Launch the authorized pilot as a detached Linux process with persistent logs."""
import json
import os
from pathlib import Path
import subprocess
import sys
from common import PROJECT,OUTPUT,dump

def launch():
    root=OUTPUT/'fold_1'
    check=json.loads((root/'preflight.json').read_text())
    if not check['passed']:
        raise RuntimeError('Preflight not passed')
    # All five two-epoch smoke runs must finish and replay their checkpoints.
    hashes=[];orders=[]
    for mode in ['F0','F1','F2','F3','F4']:
        p=root/'smoke'/f'{mode}_seed42'
        summary=json.loads((p/'summary.json').read_text())
        config=json.loads((p/'config.json').read_text())
        if not summary['checkpoint_replay_verified'] or not summary['complete']:
            raise RuntimeError('Smoke not complete '+mode)
        hashes.append(config['initial_trainable_parameters_sha256'])
        orders.append(config['first_epoch_sampler_order_sha256'])
    if len(set(hashes))!=1 or len(set(orders))!=1:
        raise RuntimeError('Unpaired smoke runs')
    if orders[0]!=check['first_epoch_sampler_order_sha256']:
        raise RuntimeError('Sampler audit mismatch')
    if (root/'suite.lock').exists() or (root/'launch.json').exists():
        raise RuntimeError('Suite already launched; inspect status instead')
    script=Path(__file__).resolve().parent
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    cmd=[sys.executable,'-B','-u',str(script/'run_suite.py'),'--fold','1','--device','cpu']
    with open(root/'suite.log','x') as log:
        child=subprocess.Popen(cmd,cwd=str(PROJECT),env=env,stdout=log,stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL,start_new_session=True)
    value=dict(pid=child.pid,command=cmd,log=str(root/'suite.log'),scope='F0-F4 x seeds42,43,44; Fold1; validation only',
        note='Frozen feature heads on one CPU thread; encoder extraction already completed on GPU0.')
    dump(root/'launch.json',value)
    print(json.dumps(value,indent=2))

if __name__=='__main__':launch()
