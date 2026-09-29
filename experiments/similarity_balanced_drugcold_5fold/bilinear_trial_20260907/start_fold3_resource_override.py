"""User-authorized resource-only restart of the waiting Fold3 worker."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path('/data1/ztx/MyModel-MDTA')
HERE = Path(__file__).resolve().parent
ROOT = PROJECT / 'outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/bilinear_rank16_trial_20260907'
TARGET = ROOT / 'fold_3'
STATUS = TARGET / 'worker_status.json'

def verify():
    for path, expected in json.loads((ROOT / 'locked_protocol.json').read_text())['sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path

def worker():
    def status(state, **extra):
        STATUS.write_text(json.dumps(dict(state=state, fold=3, gpu=1, pid=os.getpid(), timestamp=time.time(), **extra), indent=2))
    try:
        verify()
        free = int(subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        assert free >= 26000, f'Insufficient free memory at launch: {free}'
        cfg = json.loads((TARGET / 'launch_config.json').read_text())
        assert Path(cfg['output_dir']) == TARGET
        assert cfg['batch_size'] == 16 and cfg['seed'] == 42
        command = [sys.executable, '-u', str(HERE / 'train_p13d_bilinear.py')]
        for key, val in cfg.items():
            command.extend(['--' + key, str(val)])
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='1', PYTHONUNBUFFERED='1')
        with (TARGET / 'train.log').open('x') as log:
            child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
            status('training', trainer_pid=child.pid, resource_threshold_mib=26000)
            code = child.wait()
        assert code == 0, f'trainer exit={code}; inspect train.log; no automatic parameter changes'
        assert (TARGET / 'best_summary.json').is_file()
        assert (TARGET / 'best_val_predictions.npz').is_file()
        status('completed')
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise

def launch():
    verify()
    original = json.loads(STATUS.read_text())
    assert original['state'] == 'waiting_for_gpu'
    assert original['pid'] == 1377775 and original['gpu'] == 1 and original['fold'] == 3
    assert not (TARGET / 'train.log').exists()
    cmd = Path('/proc/1377775/cmdline').read_bytes().decode().split('\0')
    assert str(HERE / 'run_bilinear_trial.py') in cmd
    assert cmd[cmd.index('--fold')+1] == '3' and cmd[cmd.index('--gpu')+1] == '1'
    children = Path('/proc/1377775/task/1377775/children').read_text().strip()
    assert not children, f'Waiting worker has child processes: {children}'
    free = int(subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    assert free >= 26000
    amendment = {'reason':'User authorized attempting Fold3 with currently available GPU1 memory', 'old_threshold_mib':34000, 'new_threshold_mib':26000, 'free_mib':free, 'old_status':original, 'training_config_changed':False, 'source_hashes_verified':True, 'timestamp':time.time()}
    with (TARGET / 'resource_override.json').open('x') as f:
        json.dump(amendment, f, indent=2)
    os.kill(1377775, signal.SIGTERM)
    with (ROOT / 'worker_fold3_resource_override.log').open('x') as log:
        proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker'], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(json.dumps({'worker_pid':proc.pid, 'fold':3, 'gpu':1, 'resource_threshold_mib':26000}), flush=True)

if __name__ == '__main__':
    worker() if '--worker' in sys.argv else launch()
