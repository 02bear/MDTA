"""Snapshot Davis implementations and mechanically adapt only this new run."""
import hashlib
import json
import shutil
from pathlib import Path

P = Path('/data1/ztx/MyModel-MDTA')
H = Path(__file__).resolve().parent
assert H == P/'experiments/warm_start_811/kiba_fixed_dsrc_20260927'
assert not (H/'source').exists(), 'Never overwrite an installed experiment'
old = P/'experiments/warm_start_811/multiseed_fixed_dsrc_20260920'
pure = P/'experiments/warm_start_811/p13d_pure_20260918'
tune = P/'experiments/warm_start_811/rank16_pcim_rnc_lowcost_tune_20260917'
shutil.copytree(pure/'source', H/'source', ignore=shutil.ignore_patterns('__pycache__'))
origins = {}

def replace(text, before, after):
    assert before in text, before
    return text.replace(before, after)

for name in ['train_p13d_seed.py','run_fixed_pipeline.py','run_seed.py']:
    origin = old/name
    text = origin.read_text()
    origins[name] = {'path':str(origin),'sha256':hashlib.sha256(origin.read_bytes()).hexdigest()}
    text = text.replace('data/raw/davis/','data/raw/kiba/').replace('data/processed/davis/','data/processed/kiba/')
    text = text.replace('davis_fixed_split_811_full.json','kiba_fixed_split_811_full.json')
    text = text.replace('outputs/Refine_experiment/davis/warm_start/random_pair_811_seed42/multiseed_fixed_dsrc_20260920',
                        'outputs/Refine_experiment/kiba/warm_start/random_pair_811_seed42/p13d_pcim_rnc_dsrc_20260927')
    text = text.replace('int(os.environ["WARM_SEED"])','int(os.environ.get("WARM_SEED", "42"))')
    text = text.replace('os.environ.get("WARM_GPU", "0")','os.environ.get("WARM_GPU", "1")')
    if name=='train_p13d_seed.py':
        text=replace(text,'SOURCE = PROJECT / "experiments/warm_start_811/p13d_pure_20260918/source"','SOURCE = HERE / "source"')
        text=replace(text,'OUT = RUN_ROOT / "p13d"','RUN_ROOT = Path(os.environ.get("WARM_RUN_ROOT", str(RUN_ROOT)))\nOUT = RUN_ROOT / "p13d"')
        text=replace(text,'assert len(dataset) == 30056','assert len(dataset) == 118254')
        text=replace(text,'len(train_set) == 24044 and len(val_set) == 3005 and len(test_set) == 3007','len(train_set) == 94603 and len(val_set) == 11825 and len(test_set) == 11826')
        text=replace(text,'    model = base.build_model(args, device)', '''    import pandas as pd
    raw = pd.read_csv(PROJECT / args.pairs_csv, dtype={'drug_id':str,'protein_id':str})
    assert dataset.df[['drug_id','protein_id']].equals(raw[['drug_id','protein_id']])
    assert np.array_equal(dataset.df.label.to_numpy(), raw.label.to_numpy())
    model = base.build_model(args, device)''')
    elif name=='run_fixed_pipeline.py':
        text=replace(text,'P13D_EXPERIMENT = PROJECT / "experiments/warm_start_811/p13d_pure_20260918"',
                     'RUN_ROOT = Path(os.environ.get("WARM_RUN_ROOT", str(RUN_ROOT)))\nP13D_EXPERIMENT = Path(__file__).resolve().parent')
        text=replace(text,'TUNE_EXPERIMENT = PROJECT / "experiments/warm_start_811/rank16_pcim_rnc_lowcost_tune_20260917"','TUNE_EXPERIMENT = P13D_EXPERIMENT')
        text=replace(text,'import run_experiment as p13d_runner','import train_p13d_seed as p13d_runner')
        text=text.replace('len(frame) != 30056','len(frame) != 118254')
        text=replace(text,'if not (pair_indices >= 0).all() or np.unique(pair_indices).size != len(frame):',
                     'if np.count_nonzero(pair_indices >= 0) != len(frame) or np.unique(pair_indices[pair_indices >= 0]).size != len(frame):')
        text=text.replace('Davis pair grid is incomplete or duplicated','KIBA observed pair grid is duplicated')
        text=replace(text,'labels = np.empty_like(pair_indices, dtype=np.float64)','labels = np.full(pair_indices.shape, np.nan, dtype=np.float64)')
        text=replace(text,'base_prediction = np.empty_like(labels)','base_prediction = np.full_like(labels, np.nan)')
        text=replace(text,'np.isfinite(base_prediction).all()','np.isfinite(base_prediction[row_drug, row_protein]).all()')
        text=text.replace('[24044, 3005, 3007]','[94603, 11825, 11826]')
        text=replace(text,'"high_threshold": 7.0','"high_threshold": json.loads((P13D_EXPERIMENT / "data_audit.json").read_text())["rnc_high_threshold"]')
        text=replace(text,'"mid_threshold": 5.0','"mid_threshold": json.loads((P13D_EXPERIMENT / "data_audit.json").read_text())["rnc_mid_threshold"]')
        text=text.replace('        "reference_seed42_test_mse": 0.18321482603472625,','        "dataset": "KIBA",\n        "rnc_sampling": "training-only median and upper quartile; no label normalization",')
        text=text.replace('warm_start_p13d_pcim_rnc_fixed_dsrc_seed_','kiba_warm_start_p13d_pcim_rnc_fixed_dsrc_seed_')
    else:
        text=replace(text,'HERE = PROJECT / "experiments/warm_start_811/multiseed_fixed_dsrc_20260920"','HERE = Path(__file__).resolve().parent')
        text=text.replace('int(os.environ["WARM_GPU"])','int(os.environ.get("WARM_GPU", "1"))')
        text=replace(text,'    python = sys.executable','''    python = sys.executable
    free = int(subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True).strip())
    if free < 30000:
        raise RuntimeError(f'GPU {GPU} has only {free} MiB free')
    if not (HERE / 'smoke_result.json').exists():
        raise RuntimeError('Smoke test must pass before full training')''')
    (H/name).write_text(text)

for name in ['tune_experiment.py','pair_interaction.py','rnc_loss.py']:
    shutil.copy2(tune/name,H/name)
    origins[name]={'path':str(tune/name),'sha256':hashlib.sha256((tune/name).read_bytes()).hexdigest()}
# Correct the logged weighted objective without changing any training gradients.
f=H/'tune_experiment.py';text=f.read_text()
text=replace(text,'task.backward();rnc_value=0.0','task.backward();rnc_value=0.0;weighted_rnc=0.0')
text=replace(text,'rnc_value=float(rloss.detach());rncs.append(rnc_value)',
             "rnc_value=float(rloss.detach());weighted_rnc=cfg['rnc_weight']*cfg['interval']*warm*rnc_value;rncs.append(rnc_value)")
text=replace(text,"float(task.detach())+cfg['rnc_weight']*rnc_value","float(task.detach())+weighted_rnc")
f.write_text(text)
shutil.copy2(P/'experiments/klifs85_interaction/calibrate_sequence_fallback.py',H/'calibrate_sequence_fallback.py')
for f in H.rglob('*.py'): compile(f.read_bytes(),str(f),'exec')
(H/'source_manifest.json').write_text(json.dumps({'origins':origins,'installed_sha256':{str(f.relative_to(H)):hashlib.sha256(f.read_bytes()).hexdigest() for f in H.rglob('*.py')}},indent=2))
print('INSTALLED',H,flush=True)
