import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT = Path(os.environ.get('MDTA_PROJECT', '/data1/ztx/MyModel-MDTA')).resolve()
sys.path.insert(0, str(PROJECT))
OUTPUT = PROJECT/'outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/dampe_frozen_ot_20260908'
BASE = PROJECT/'outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/baseline'
SPLITS = PROJECT/'data/splits/davis_drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final'
SOURCE_FILES = ['models/model_p13d.py', 'models/fusion.py', 'models/decoder.py',
                'models/drug_1d_encoder.py', 'models/protein_1d_encoder.py',
                'models/drug_3d_egnn_encoder.py', 'models/protein_3d_egnn_encoder.py',
                'datasets/davis_dataset_p13d.py', 'datasets/collate_p13d.py',
                'train_p13d_earlystop.py']

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)

def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else PROJECT/p

def source_hashes():
    return {f: sha256(PROJECT/f) for f in SOURCE_FILES}

def assert_sources(cache):
    if source_hashes() != cache['source_sha256']:
        raise RuntimeError('Baseline source changed since cache creation')

def validate_split(df, split):
    groups = {}
    drugsets = {}
    for part in ['train', 'val', 'test']:
        ix = split[part+'_indices']
        if not ix or len(ix) != len(set(ix)) or min(ix) < 0 or max(ix) >= len(df):
            raise ValueError('Invalid '+part+' indices')
        groups[part] = set(ix)
        drugsets[part] = set(df.iloc[ix].drug_id.astype(str))
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if groups[a]&groups[b] or drugsets[a]&drugsets[b]:
            raise ValueError('Pair/drug overlap between '+a+' and '+b)
    if set.union(*groups.values()) != set(range(len(df))):
        raise ValueError('Split does not partition filtered dataset')
    return {part: {'pairs': len(groups[part]), 'drugs': len(drugsets[part]),
                   'proteins': int(df.iloc[split[part+'_indices']].protein_id.nunique())}
            for part in groups}
