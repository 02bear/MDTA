"""Detached per-GPU workers. Wait for sufficient free memory, never kill other jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path('/data1/ztx/MyModel-MDTA')
HERE = Path(__file__).resolve().parent
ROOT = PROJECT / 'outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/bilinear_rank16_trial_20260907'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def worker(fold, gpu):
    import torch
    target = ROOT / f'fold_{fold}'
    target.mkdir(exist_ok=False)
    status = target / 'worker_status.json'
    def write(state, **kw):
        status.write_text(json.dumps(dict(state=state, fold=fold, gpu=gpu, pid=os.getpid(), timestamp=time.time(), **kw), indent=2))
    write('waiting_for_gpu')
    try:
        manifest = json.loads((ROOT / 'locked_protocol.json').read_text())
        for path, sha in manifest['sha256'].items():
            assert digest(Path(path)) == sha, f'Source changed: {path}'
        ck = torch.load(PROJECT / f'outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/baseline/fold_{fold}/best_model.pt', map_location='cpu', weights_only=False)
        config = ck['args'].copy()
        del ck
        config['output_dir'] = str(target)
        (target / 'launch_config.json').write_text(json.dumps(config, indent=2))
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            if free >= 34000:
                break
            write('waiting_for_gpu', free_mib=free, required_mib=34000)
            time.sleep(60)
        for path, sha in manifest['sha256'].items():
            assert digest(Path(path)) == sha, f'Source changed while queued: {path}'
        command = [sys.executable, '-u', str(HERE / 'train_p13d_bilinear.py')]
        for key, value in config.items():
            command.extend(['--' + key, str(value)])
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['PYTHONUNBUFFERED'] = '1'
        with (target / 'train.log').open('x') as log:
            child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write('training', trainer_pid=child.pid)
            code = child.wait()
        assert code == 0, f'trainer exit={code}'
        assert (target / 'best_summary.json').is_file()
        assert (target / 'best_val_predictions.npz').is_file()
        write('completed')
    except BaseException as exc:
        write('failed', error=repr(exc))
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--fold', type=int, choices=[1, 3])
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if not args.launch:
        return worker(args.fold, args.gpu)
    ROOT.mkdir(exist_ok=False)
    sources = [PROJECT / 'train_p13d_earlystop.py', *HERE.glob('*.py')]
    sources += list((PROJECT / 'models').glob('*.py')) + list((PROJECT / 'datasets').glob('*p13d*.py'))
    sources += [PROJECT / f'data/splits/davis_drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/fold_{f}/split.json' for f in (1, 3)]
    (ROOT / 'locked_protocol.json').write_text(json.dumps({'head': 'a(d)+b(p)+dot(Ud,Vp)/sqrt(16)', 'rank':16, 'seed':42, 'folds':[1,3], 'fresh_start':True, 'baseline_weights_loaded_into_model':False, 'all_other_args':'copied from each baseline checkpoint args', 'success':'development: both folds MSE below baseline; CI/Rm2 and per-drug/strong-affinity errors as consistency checks; nonzero interaction alone not success', 'sha256':{str(p):digest(p) for p in sources}}, indent=2))
    for fold, gpu in [(1, 0), (3, 1)]:
        with (ROOT / f'worker_fold{fold}.log').open('x') as log:
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--fold', str(fold), '--gpu', str(gpu)], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, stdin=subprocess.DEVNULL)
        print(json.dumps({'fold':fold, 'gpu':gpu, 'worker_pid':proc.pid, 'root':str(ROOT)}), flush=True)

if __name__ == '__main__':
    main()
