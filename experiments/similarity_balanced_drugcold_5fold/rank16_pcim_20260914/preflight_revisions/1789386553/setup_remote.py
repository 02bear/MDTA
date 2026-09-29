"""Install isolated dependencies and pre-register protocol; never edit historical experiments."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from common import HERE, OLD, OUT, BASE_OUT, SPLITS, sha, dump

assert not (HERE/'manifest.json').exists(), 'Already installed; do not overwrite a locked run'
old_manifest=json.loads((OLD/'manifest.json').read_text())
for path,expected in old_manifest['sha256'].items():
    assert sha(path)==expected,path
shutil.copytree(OLD/'source',HERE/'source',ignore=shutil.ignore_patterns('__pycache__'))
for name in ['model_impl.py','r1_source.py','entity_similarities.npz','r1_reference_summary.json']:
    shutil.copyfile(OLD/name,HERE/name)
(HERE/'references').mkdir(exist_ok=False)
external=[]
for fold in range(1,6):
    source=BASE_OUT/'bilinear_rank16_complete_20260914'/f'evaluation/fold_{fold}/results.json'
    ref=json.loads(source.read_text())
    assert sha(ref['checkpoint'])==ref['checkpoint_sha256']
    assert Path(ref['split']).resolve()==(SPLITS/f'fold_{fold}/split.json').resolve()
    assert sha(ref['split'])==ref['split_sha256']
    shutil.copyfile(source,HERE/f'references/fold_{fold}.json')
    external += [Path(ref['checkpoint']),Path(ref['split']),source,
                 source.parent/'train_predictions.npz',source.parent/'test_predictions.csv',
                 Path(ref['checkpoint']).parent/'best_val_predictions.npz']
cfg={'name':'rank16_pcim_20260914','split':'davis_drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final',
     'folds':[1,2,3,4,5],'seeds':[42,2026,3407],'backbone_seed':42,
     'variants':['pair_graph','pair_pool','global_mlp'],'primary_variant':'pair_graph',
     'width':32,'top_k':64,'heads':2,'graph_layers':1,'condition_dim':16,'residue_type_dim':8,
     'max_delta':.5,'dropout':.1,'batch_size':16,'eval_batch_size':64,
     'epochs':60,'patience':10,'min_delta':1e-5,'lr':3e-4,'weight_decay':1e-5,
     'grad_clip':1.0,'residual_l2':1e-3,'lambdas':[0,.25,.5,1.0],
     'optimizer':'AdamW','loss':'MSE(frozen_rank16 + raw_delta, train_y) + 1e-3*mean(delta^2)',
     'selection':'minimum validation MSE of combined model + recomputed R1; lambda 0 is baseline',
     'r1':'locked historical per-fold hyperparameters, residuals recomputed for each checkpoint and lambda',
     'schedule':'cache all folds; primary model 3 seeds x 5 folds; pair_pool controls; global_mlp controls',
     'gpu':0,'frozen_backbone_eval':True,'source_feature_cache':'float32 per-entity node/global features',
     'test_policy':'score only after per-run checkpoint and lambda are locked; no adaptive test-based selection',
     'interpretation':'latent compatibility, not physical contact; independent ligand/protein coordinates never mixed',
     'evidence_limit':'same repeatedly inspected development split; varying adapter seeds does not vary backbone seeds'}
dump(HERE/'protocol.json',cfg)
files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
external += [Path('/data1/ztx/MyModel-MDTA/data/raw/davis/pairs.csv')]
dump(HERE/'manifest.json',{'sha256':{str(p):sha(p) for p in files+external},'historical_sources_verified':True})
OUT.mkdir(exist_ok=False)
(OUT/'logs').mkdir()
shutil.copyfile(HERE/'protocol.json',OUT/'protocol.json')
shutil.copyfile(HERE/'manifest.json',OUT/'provenance.json')
for p in HERE.rglob('*.py'):compile(p.read_bytes(),str(p),'exec')
print(json.dumps({'installed':True,'code':str(HERE),'output':str(OUT),'protocol':cfg},indent=2))
