"""Locked paths and shared utilities for end-to-end Rank16 unfreezing."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT = Path('/data1/ztx/MyModel-MDTA')
BASE_OUT = PROJECT/'outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final'
PARENT = PROJECT/'experiments/similarity_balanced_drugcold_5fold/rank16_pcim_pair_rnc_20260915'
PARENT_OUT = BASE_OUT/'rank16_pcim_pair_rnc_20260915'
CACHE_SOURCE = BASE_OUT/'rank16_pcim_20260914/cache'
RANK_OUT = BASE_OUT/'bilinear_rank16_complete_20260914'
OUT = BASE_OUT/'rank16_pcim_pair_rnc_unfreeze_20260916'
SPLITS = PROJECT/'data/splits/davis_drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final'


def sha(path):
    dig = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            dig.update(block)
    return dig.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:
        torch.save(obj, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def locked_modules():
    source = HERE/'source'
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('unfreeze_locked_rank16', HERE/'model_impl.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.base.__file__).resolve() == source/'train_p13d_earlystop.py'
    return module, module.base


def verify_manifest():
    manifest = json.loads((HERE/'manifest.json').read_text())
    for path, expected in manifest['sha256'].items():
        assert sha(path) == expected, f'Locked input changed: {path}'
    return manifest


def load_protocol():
    return json.loads((HERE/'protocol.json').read_text())


def r1_functions():
    tree = ast.parse((HERE/'r1_source.py').read_text())
    scope = {'np': np}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {'top_weights','drug_kernel_stats','apply_correction'}:
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<locked-r1>', 'exec'), scope)
    return scope


class R1:
    """Only training labels are available to the residual corrector."""
    def __init__(self, cache):
        self.train = np.asarray(cache['split_drugs']['train'])
        self.y_train = cache['labels'][self.train].astype(np.float64)
        self.config = cache['r1_config']
        self.functions = r1_functions()
        self.sim = cache['similarity'].astype(np.float64)

    def correct(self, prediction, query):
        assert np.isfinite(prediction[self.train]).all()
        residual = self.y_train - prediction[self.train]
        mean, var, support = self.functions['drug_kernel_stats'](
            np.asarray(query), self.train, residual, self.sim, self.config)
        return self.functions['apply_correction'](prediction[query], mean, var, support, self.config)


def metric(prediction, label):
    _, base = locked_modules()
    return base.compute_regression_metrics(
        torch.tensor(np.asarray(prediction).ravel(), dtype=torch.float64),
        torch.tensor(np.asarray(label).ravel(), dtype=torch.float64))
