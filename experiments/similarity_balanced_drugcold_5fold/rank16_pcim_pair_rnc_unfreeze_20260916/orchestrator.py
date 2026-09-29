"""Run two independent preflights and workers on physical GPUs 0 and 1."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from common import HERE, OUT, PROJECT, dump, load_protocol, verify_manifest


def free_mib(gpu):
    return int(subprocess.check_output(
        ['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],
        text=True).strip())


def launch(command, gpu, logpath):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    log = logpath.open('a')
    process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               start_new_session=True)
    return process, log


def main():
    verify_manifest(); cfg = load_protocol()
    variants = cfg['variants']; assignment = cfg['gpu_assignment']
    dump(OUT/'orchestrator_status.json', {'state':'preflight_starting','pid':os.getpid(),'updated':time.time()})
    for variant in variants:
        gpu = assignment[variant]
        free = free_mib(gpu)
        assert free >= cfg['minimum_gpu_free_mib'], (variant, gpu, free)
    running = {}
    for variant in variants:
        gpu = assignment[variant]
        command = [sys.executable,'-B','-u',str(HERE/'preflight.py'),'--variant',variant]
        process, log = launch(command, gpu, OUT/f'logs/preflight_{variant}.log')
        running[variant] = (process, log, gpu)
    for variant, (process, log, gpu) in running.items():
        code = process.wait(); log.close()
        if code:
            raise RuntimeError(f'preflight {variant} failed on GPU {gpu} with exit {code}')
        result = json.loads((OUT/f'preflight_{variant}.json').read_text())
        assert result['passed']
    dump(OUT/'orchestrator_status.json', {'state':'workers_starting','pid':os.getpid(),'updated':time.time()})
    running = {}
    for variant in variants:
        gpu = assignment[variant]
        command = [sys.executable,'-B','-u',str(HERE/'train_experiment.py'),'worker','--variant',variant]
        process, log = launch(command, gpu, OUT/f'logs/worker_{variant}.log')
        running[variant] = (process, log, gpu)
    dump(OUT/'orchestrator_status.json', {'state':'running','pid':os.getpid(),
         'workers':{v:{'pid':x[0].pid,'gpu':x[2]} for v,x in running.items()},'updated':time.time()})
    failures = []
    for variant, (process, log, gpu) in running.items():
        code = process.wait(); log.close()
        if code: failures.append({'variant':variant,'gpu':gpu,'exit':code})
    if failures:
        raise RuntimeError(f'worker failures: {failures}')
    subprocess.check_call([sys.executable,'-B','-u',str(HERE/'train_experiment.py'),'aggregate'], cwd=PROJECT)
    dump(OUT/'orchestrator_status.json', {'state':'completed','pid':os.getpid(),'finished':time.time()})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        dump(OUT/'orchestrator_status.json', {'state':'failed','pid':os.getpid(),
             'error':str(exc),'traceback':traceback.format_exc(),'updated':time.time()})
        raise
