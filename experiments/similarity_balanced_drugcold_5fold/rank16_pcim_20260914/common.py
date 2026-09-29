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
OLD = PROJECT/'experiments/similarity_balanced_drugcold_5fold/rank16_complete_20260914'
BASE_OUT = PROJECT/'outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final'
OUT = BASE_OUT/'rank16_pcim_20260914'
SPLITS = PROJECT/'data/splits/davis_drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final'


def sha(path):
    dig = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): dig.update(block)
    return dig.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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


def restore_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    torch.cuda.set_rng_state_all(s['cuda'])


def locked_modules():
    sys.path.insert(0, str(HERE/'source'))
    spec = importlib.util.spec_from_file_location('locked_rank16', HERE/'model_impl.py')
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    assert Path(h.base.__file__).resolve() == HERE/'source/train_p13d_earlystop.py'
    return h, h.base


def verify_manifest():
    obj = json.loads((HERE/'manifest.json').read_text())
    for p, expected in obj['sha256'].items():
        assert sha(p) == expected, f'Locked input changed: {p}'
    return obj


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
    """Only constructor receives training labels; query labels never enter prediction."""
    def __init__(self, cache):
        self.train = np.array(cache['split_drugs']['train'])
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


class FeatureStore:
    def __init__(self, cache, device='cuda'):
        from torch.nn.utils.rnn import pad_sequence
        self.device = device
        self.atom_length = [len(x) for x in cache['atoms']]
        self.protein_length = [len(x) for x in cache['residues']]
        self.atoms = pad_sequence(cache['atoms'], batch_first=True).to(device)
        self.residues = pad_sequence(cache['residues'], batch_first=True).to(device)
        self.aa = pad_sequence(cache['aa'], batch_first=True, padding_value=20).to(device)
        self.drug_global = cache['drug_global'].to(device)
        self.protein_global = cache['protein_global'].to(device)
        self.atom_mask = torch.arange(self.atoms.size(1), device=device)[None] < torch.tensor(self.atom_length, device=device)[:, None]
        self.residue_mask = torch.arange(self.residues.size(1), device=device)[None] < torch.tensor(self.protein_length, device=device)[:, None]

    def batch(self, drug, protein, perturb=None, global_only=False):
        d = torch.as_tensor(drug, device=self.device)
        p = torch.as_tensor(protein, device=self.device)
        b = {'drug_global': self.drug_global[d], 'protein_global': self.protein_global[p]}
        if global_only: return b
        # Controlled mismatch changes local ligand identity but preserves global identity.
        local_d = (d+1) % len(self.atom_length) if perturb == 'mismatch_drug_local' else d
        nd = max(self.atom_length[int(x)] for x in (np.asarray(drug)+1) % len(self.atom_length)) if perturb == 'mismatch_drug_local' else max(self.atom_length[int(x)] for x in drug)
        nr = max(self.protein_length[int(x)] for x in protein)
        b.update(atoms=self.atoms[local_d, :nd], atom_mask=self.atom_mask[local_d, :nd],
                 residues=self.residues[p, :nr], residue_mask=self.residue_mask[p, :nr], aa=self.aa[p, :nr])
        if perturb == 'shuffle_aa':
            b['aa'] = b['aa'].clone()
            for i, pid in enumerate(protein):
                n = self.protein_length[int(pid)]
                b['aa'][i, :n] = b['aa'][i, :n].roll(7)
        return b


def metric(prediction, label):
    _, t = locked_modules()
    return t.compute_regression_metrics(torch.tensor(np.asarray(prediction).ravel(), dtype=torch.float64),
                                        torch.tensor(np.asarray(label).ravel(), dtype=torch.float64))
