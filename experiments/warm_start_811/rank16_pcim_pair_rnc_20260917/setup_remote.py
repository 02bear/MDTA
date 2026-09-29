"""Install immutable inputs and protocol for the pair-level RNC stage-1 screen."""
import json
import shutil

from common import HERE, PARENT, PARENT_OUT, CACHE_SOURCE, OUT, SPLITS, sha, dump


assert not (HERE/'manifest.json').exists()
assert not OUT.exists(), 'Never overwrite a result directory'
parent_manifest=json.loads((PARENT/'manifest.json').read_text())
for path,expected in parent_manifest['sha256'].items():
    assert sha(path)==expected,path
summary=json.loads((PARENT_OUT/'summary.json').read_text())
assert summary['complete'] and summary['completed_runs']==60

shutil.copytree(PARENT/'source',HERE/'source',ignore=shutil.ignore_patterns('__pycache__'))
for name in ['model_impl.py','r1_source.py','entity_similarities.npz','r1_reference_summary.json']:
    shutil.copyfile(PARENT/name,HERE/name)

cfg=json.loads((PARENT/'protocol.json').read_text())
cfg.update(
    name='rank16_pcim_pair_rnc_20260915',
    variants=['baseline','rnc_standard_a001','rnc_standard_a003',
              'rnc_high_a001','rnc_high_a003'],
    variant_specs={
        'baseline':{'rnc_mode':'none','rnc_weight':0.0},
        'rnc_standard_a001':{'rnc_mode':'standard','rnc_weight':0.01},
        'rnc_standard_a003':{'rnc_mode':'standard','rnc_weight':0.03},
        'rnc_high_a001':{'rnc_mode':'high_weighted','rnc_weight':0.01},
        'rnc_high_a003':{'rnc_mode':'high_weighted','rnc_weight':0.03},
    },
    primary_variant='rnc_high_a001',
    seeds=[42],
    minimum_gpu_free_mib=12000,
    gpu_sharing_authorized=True,
    schedule='Stage 1: five RNC configurations x seed 42 x five drug-cold folds = 25 fresh runs',
    source_feature_cache=str(CACHE_SOURCE),
    historical_controls=str(PARENT_OUT),
    candidate_selection='minimum validation MSE after combined prediction and rebuilt R1; lambda 0 remains the baseline boundary',
    loss='MSE(frozen_rank16 + raw_delta, train_y) + 1e-3*mean(delta^2) + scheduled pair-level RNC',
    rnc={
        'representation':'16D L2-normalized projection of the 32D conditioned PairGraph pooled vector z',
        'labels':'full observed pKd affinity from the training partition only',
        'projection':[32,32,16],
        'projection_dim':16,
        'temperature':2.0,
        'batch_size':32,
        'microbatch_size':16,
        'interval':2,
        'high_count':8,
        'mid_count':8,
        'high_threshold':7.0,
        'mid_threshold':5.0,
        'high_width':0.5,
        'high_strength':2.0,
        'warmup_epochs':1.0,
        'sampling':'8 y>=7, 8 5<y<7, then 16 distinct remaining training pairs; shortages filled from remaining pairs',
        'standard':'original continuous-label Rank-N-Contrast ordering',
        'high_weighted':'same RNC ordering; anchor weight 1+2*sigmoid((y-7)/0.5)',
        'inference':'projection head is unused at inference',
    },
)
assert cfg['top_k']==64 and cfg['seeds']==[42] and cfg['gpu']==0
assert cfg['coverage_budget']==32 and cfg['cross_attention']['stage'].startswith('after frozen Rank16')

external=[]
for fold in range(1,6):
    cache=CACHE_SOURCE/f'fold_{fold}.pt';audit=CACHE_SOURCE/f'fold_{fold}_audit.json'
    obj=json.loads(audit.read_text());assert obj['passed'] and sha(cache)==obj['cache_sha256']
    external += [cache,audit,SPLITS/f'fold_{fold}/split.json']
    run=PARENT_OUT/f'runs/bidirectional/seed_42/fold_{fold}'
    result=json.loads((run/'result.json').read_text())
    assert sha(run/'best.pt')==result['adapter_sha256']
    external += [run/'result.json',run/'history.json',run/'best.pt',run/'selection_locked.json']
external += [PARENT/'pair_interaction.py',PARENT/'protocol.json',PARENT/'manifest.json',PARENT_OUT/'summary.json']

dump(HERE/'protocol.json',cfg)
files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
dump(HERE/'manifest.json',{'sha256':{str(p):sha(p) for p in files+external},
                          'parent':str(PARENT),'controls':str(PARENT_OUT)})
OUT.mkdir();(OUT/'logs').mkdir()
dump(OUT/'historical_controls.json',{
    'source':str(PARENT_OUT),
    'group':summary['groups']['bidirectional']['42'],
    'runs':5,
    'note':'Context only. The stage-1 baseline is rerun contemporaneously.',
})
shutil.copyfile(HERE/'protocol.json',OUT/'protocol.json')
shutil.copyfile(HERE/'manifest.json',OUT/'provenance.json')
for p in HERE.rglob('*.py'):
    compile(p.read_bytes(),str(p),'exec')
print(json.dumps({'installed':True,'code':str(HERE),'output':str(OUT),
                  'new_runs':25,'protocol':cfg},indent=2))
