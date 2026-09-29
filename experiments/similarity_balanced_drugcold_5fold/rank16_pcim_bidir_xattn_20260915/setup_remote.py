"""Lock protocol, parent controls, and frozen caches before starting training."""
import json
import shutil
from common import HERE, PARENT, PARENT_OUT, CACHE_SOURCE, OUT, SPLITS, sha, dump

assert not (HERE/'manifest.json').exists()
assert not OUT.exists(), 'Never overwrite a result directory'
parent_manifest=json.loads((PARENT/'manifest.json').read_text())
for path,expected in parent_manifest['sha256'].items():
    assert sha(path)==expected,path
summary=json.loads((PARENT_OUT/'summary.json').read_text())
assert summary['complete'] and summary['completed_runs']==45
shutil.copytree(PARENT/'source',HERE/'source',ignore=shutil.ignore_patterns('__pycache__'))
for name in ['model_impl.py','r1_source.py','entity_similarities.npz','r1_reference_summary.json']:
    shutil.copyfile(PARENT/name,HERE/name)
cfg=json.loads((PARENT/'protocol.json').read_text())
cfg.pop('conditional_attention',None)
cfg.pop('revision_reason',None)
cfg.update(name='rank16_pcim_bidir_xattn_20260915',
           variants=['bidirectional','atom_conditioned','residue_conditioned','baseline'],
           primary_variant='bidirectional',
           candidate_selection='coverage',coverage_budget=32,
           schedule='60 fresh runs: bidirectional/two one-way controls/baseline x three adapter seeds x five folds',
           source_feature_cache=str(CACHE_SOURCE),historical_controls=str(PARENT_OUT),
           cross_attention={
             'stage':'after frozen Rank16 local projection and before coverage Top-64 pair selection',
             'atom_source':'drug_3d_encoder(..., return_node=True)[node_feat], 128D then projected to 32D',
             'residue_source':'protein_3d_encoder(..., return_node=True)[node_feat], 128D plus 8D residue type then projected to 32D',
             'global_features':'excluded from new cross-attention; retained only in the existing gate and pool',
             'directions':['atoms query residues and update atoms','residues query atoms and update residues'],
             'width':32,'heads':2,'max_residual_scale':0.25,'extra_parameters':8578,
             'gate':'0.25*tanh(gamma) per direction; gamma initialized exactly zero',
             'controls':['atom_conditioned','residue_conditioned','cross off','opposite entity shuffled'],
             'diagnostics':['directional entropy','update RMS and ratio','gate values','Top-64 Jaccard','rebuilt R1 validation MSE']})
assert cfg['top_k']==64 and cfg['seeds']==[42,2026,3407] and cfg['gpu']==0
external=[]
for fold in range(1,6):
    cache=CACHE_SOURCE/f'fold_{fold}.pt';audit=CACHE_SOURCE/f'fold_{fold}_audit.json'
    obj=json.loads(audit.read_text());assert obj['passed'] and sha(cache)==obj['cache_sha256']
    external += [cache,audit,SPLITS/f'fold_{fold}/split.json']
controls={'baseline':summary['groups']['baseline']}
for seed in cfg['seeds']:
    for fold in cfg['folds']:
        run=PARENT_OUT/f'runs/baseline/seed_{seed}/fold_{fold}'
        obj=json.loads((run/'result.json').read_text())
        assert sha(run/'best.pt')==obj['adapter_sha256']
        external += [run/'result.json',run/'history.json',run/'best.pt',run/'selection_locked.json']
external += [PARENT/'pair_interaction.py',PARENT/'protocol.json',PARENT/'manifest.json']
external += [PARENT_OUT/'summary.json']
dump(HERE/'protocol.json',cfg)
files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
dump(HERE/'manifest.json',{'sha256':{str(p):sha(p) for p in files+external},
                          'parent':str(PARENT),'controls':str(PARENT_OUT)})
OUT.mkdir()
(OUT/'logs').mkdir()
dump(OUT/'historical_controls.json',{'source':str(PARENT_OUT),'groups':controls,'runs':15,
     'note':'Historical final baseline is context only; all 60 directional comparison runs are fresh.'})
shutil.copyfile(HERE/'protocol.json',OUT/'protocol.json')
shutil.copyfile(HERE/'manifest.json',OUT/'provenance.json')
for p in HERE.rglob('*.py'):compile(p.read_bytes(),str(p),'exec')
print(json.dumps({'installed':True,'code':str(HERE),'output':str(OUT),'new_runs':60,'protocol':cfg},indent=2))
