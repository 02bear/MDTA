"""Install immutable inputs and protocol for the two-branch unfreezing screen."""
import json
import shutil

from common import (CACHE_SOURCE, HERE, OUT, PARENT, PARENT_OUT, SPLITS,
                    dump, sha)


assert not (HERE/'manifest.json').exists()
assert not OUT.exists(), 'Never overwrite a result directory'
parent_manifest = json.loads((PARENT/'manifest.json').read_text())
for path, expected in parent_manifest['sha256'].items():
    assert sha(path) == expected, path
parent_summary = json.loads((PARENT_OUT/'summary.json').read_text())
assert parent_summary['complete'] and parent_summary['completed_runs'] == 25

shutil.copytree(PARENT/'source', HERE/'source', ignore=shutil.ignore_patterns('__pycache__'))
for name in ['model_impl.py','r1_source.py','entity_similarities.npz','r1_reference_summary.json']:
    shutil.copyfile(PARENT/name, HERE/name)

cfg = json.loads((PARENT/'protocol.json').read_text())
cfg.update(
    name='rank16_pcim_pair_rnc_unfreeze_20260916',
    variants=['unfreeze_last','unfreeze_full'],
    seeds=[42],
    primary_variant='unfreeze_full',
    schedule='Two GPUs: last-block broad unfreezing vs progressive full unfreezing; seed 42 x five folds each',
    epochs=60,
    batch_size=16,
    eval_batch_size=16,
    evaluation_interval=2,
    early_stop_start_epoch=6,
    patience_evaluations=6,
    min_delta=1e-5,
    weight_decay=1e-5,
    grad_clip=1.0,
    residual_l2=1e-3,
    rnc_weight=0.01,
    lambdas=[0.0,0.25,0.5,1.0],
    minimum_gpu_free_mib=35000,
    gpu_assignment={'unfreeze_last':0,'unfreeze_full':1},
    learning_rates={
        'interaction':3e-4,
        'rank_head_fusion_out':3e-5,
        'egnn_last':1e-5,
        'one_d':5e-6,
        'egnn_early':5e-6,
    },
    unfreeze_schedule={
        'epochs_1_2':'warm-started PCIM/cross-attention/RNC only; Rank16 eval and frozen',
        'epochs_3_5':'Rank16 low-rank head, both fusions, both 1D encoders, both 3D out projections and third EGNN layers',
        'epoch_6_plus_last':'same broad last-block set remains trainable',
        'epoch_6_plus_full':'all Rank16 parameters trainable',
    },
    candidate_selection='validation MSE after rebuilt R1 over locked Rank16, locked frozen pair-RNC, fine-tuned Rank16, and fine-tuned Rank16+lambda*delta',
    loss='MSE(finetuned_rank16 + delta, y) + 0.01 scheduled standard RNC + 1e-3 mean(delta^2)',
)
assert cfg['rnc']['temperature'] == 2.0 and cfg['rnc']['batch_size'] == 32
assert cfg['rnc']['interval'] == 2 and cfg['rnc']['microbatch_size'] == 16

external = [PARENT/'manifest.json', PARENT/'protocol.json', PARENT/'pair_interaction.py',
            PARENT/'rnc_loss.py', PARENT_OUT/'summary.json']
adapter_initialization = {}
for fold in range(1,6):
    cache = CACHE_SOURCE/f'fold_{fold}.pt'
    audit = CACHE_SOURCE/f'fold_{fold}_audit.json'
    audit_obj = json.loads(audit.read_text())
    assert audit_obj['passed'] and sha(cache) == audit_obj['cache_sha256']
    cached = __import__('torch').load(cache, map_location='cpu', weights_only=False)
    assert sha(cached['checkpoint']) == cached['checkpoint_sha256']
    run = PARENT_OUT/f'runs/rnc_standard_a001/seed_42/fold_{fold}'
    result = json.loads((run/'result.json').read_text())
    adapter = run/('best_active.pt' if result['baseline_fallback'] else 'best.pt')
    adapter_initialization[str(fold)] = {
        'path':str(adapter), 'sha256':sha(adapter),
        'reason':'best_active keeps a trained RNC representation when the final stage-1 selector fell back' if result['baseline_fallback'] else 'stage-1 selected best checkpoint'}
    external += [cache, audit, cached['checkpoint'], SPLITS/f'fold_{fold}/split.json',
                 run/'result.json', run/'best.pt', run/'best_active.pt',
                 run/'selection_locked.json', run/'test_predictions.csv']
    if not result['baseline_fallback']:
        external.append(run/'best_development_predictions.npz')
cfg['adapter_initialization'] = adapter_initialization

dump(HERE/'protocol.json', cfg)
files = [p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
dump(HERE/'manifest.json', {'sha256':{str(p):sha(p) for p in files+external},
                            'parent':str(PARENT), 'controls':str(PARENT_OUT)})
OUT.mkdir(); (OUT/'logs').mkdir()
dump(OUT/'historical_controls.json', {
    'source':str(PARENT_OUT),
    'frozen_pair_rnc_seed42':parent_summary['groups']['rnc_standard_a001']['42'],
    'rank16_R1_mse':parent_summary['groups']['rnc_standard_a001']['42']['macro']['rank16_R1']['mse'],
    'note':'Locked candidates are evaluated without modification and remain available per fold.'})
shutil.copyfile(HERE/'protocol.json', OUT/'protocol.json')
shutil.copyfile(HERE/'manifest.json', OUT/'provenance.json')
for path in HERE.rglob('*.py'):
    compile(path.read_bytes(), str(path), 'exec')
print(json.dumps({'installed':True, 'code':str(HERE), 'output':str(OUT),
                  'new_runs':10, 'protocol':cfg}, indent=2))
