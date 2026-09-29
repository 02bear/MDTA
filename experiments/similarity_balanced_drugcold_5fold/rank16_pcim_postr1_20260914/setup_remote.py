"""Install an isolated post-R1 residual experiment from locked prior artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
from common import HERE, OLD, OLD_OUT, CACHE_SOURCE, OUT, SPLITS, sha, dump

assert not (HERE/'manifest.json').exists(), 'Already installed; do not overwrite a locked run'
old_manifest=json.loads((OLD/'manifest.json').read_text())
for path,expected in old_manifest['sha256'].items(): assert sha(path)==expected,path
old_summary=json.loads((OLD_OUT/'summary.json').read_text())
assert old_summary['complete'] and old_summary['completed_runs']==45
shutil.copytree(OLD/'source',HERE/'source',ignore=shutil.ignore_patterns('__pycache__'))
for name in ['model_impl.py','r1_source.py','entity_similarities.npz','r1_reference_summary.json']:
    shutil.copyfile(OLD/name,HERE/name)
external=[]
for fold in range(1,6):
    cache=CACHE_SOURCE/f'fold_{fold}.pt'; audit=CACHE_SOURCE/f'fold_{fold}_audit.json'
    audit_obj=json.loads(audit.read_text())
    assert audit_obj['passed'] and sha(cache)==audit_obj['cache_sha256']
    external += [cache,audit,SPLITS/f'fold_{fold}/split.json']
cfg={'name':'rank16_pcim_postr1_20260914','split':'davis_drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final',
     'folds':[1,2,3,4,5],'seeds':[42,2026,3407],'backbone_seed':42,
     'variants':['pair_graph','pair_pool','global_mlp'],'primary_variant':'pair_graph',
     'width':32,'top_k':64,'heads':2,'graph_layers':1,'condition_dim':16,'residue_type_dim':8,
     'max_delta':.5,'dropout':.1,'batch_size':16,'eval_batch_size':64,
     'epochs':60,'patience':10,'min_delta':1e-5,'lr':3e-4,'weight_decay':1e-5,
     'grad_clip':1.0,'residual_l2':1e-3,'lambdas':[0,.25,.5,1.0],
     'optimizer':'AdamW','loss':'MSE(raw_delta, train_y - leave-one-drug-out Rank16+R1) + 1e-3*mean(delta^2)',
     'selection':'minimum validation MSE of Rank16+R1 + lambda*delta; lambda 0 is exact baseline',
     'r1':'locked historical per-fold hyperparameters; training target uses leave-one-drug-out R1 references',
     'schedule':'3 variants x 3 adapter seeds x 5 folds; pair_graph runs first',
     'gpu':0,'frozen_backbone_eval':True,'source_feature_cache':str(CACHE_SOURCE),
     'test_policy':'score only after per-run checkpoint and lambda are locked; no adaptive test-based selection',
     'interpretation':'latent atom-residue compatibility residual after Rank16+R1, not physical contact',
     'evidence_limit':'same repeatedly inspected development split; varying adapter seeds does not vary backbone seeds'}
dump(HERE/'protocol.json',cfg)
files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
dump(HERE/'manifest.json',{'sha256':{str(p):sha(p) for p in files+external},
     'historical_sources_verified':True,'cache_source':str(CACHE_SOURCE),'parent_experiment':str(OLD)})
OUT.mkdir(exist_ok=False)
(OUT/'logs').mkdir()
shutil.copyfile(HERE/'protocol.json',OUT/'protocol.json')
shutil.copyfile(HERE/'manifest.json',OUT/'provenance.json')
for p in HERE.rglob('*.py'):compile(p.read_bytes(),str(p),'exec')
print(json.dumps({'installed':True,'code':str(HERE),'output':str(OUT),'protocol':cfg},indent=2))
