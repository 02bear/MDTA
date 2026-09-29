"""Resume only our test evaluator, leaving all existing training untouched."""
import json, os, signal, subprocess, sys
from pathlib import Path
ROOT=Path('/data1/ztx/MyModel-MDTA')
OUT=ROOT/'outputs/Refine_experiment/davis/cold_start/drug_cold_5fold_seed42/evaluation/missing_tests_20260908'
SCRIPT=ROOT/'experiments/similarity_balanced_drugcold_5fold/evaluate_missing_tests.py'
old=2449406
procpath=Path(f'/proc/{old}/cmdline')
if procpath.exists():
    cmd=procpath.read_bytes()
    assert b'evaluate_missing_tests.py' in cmd and b'--worker' in cmd
    os.kill(old,signal.SIGTERM)
workers=[]
for method,gpu in [('strictfp','1'),('threegrain','0'),('protected_overall','3'),('protected_robust','2')]:
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with (OUT/f'evaluate_{method}.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'--worker','--method',method],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    workers.append(dict(method=method,gpu=gpu,pid=p.pid))
(OUT/'parallel_workers.json').write_text(json.dumps(workers,indent=2))
print(json.dumps(workers))
