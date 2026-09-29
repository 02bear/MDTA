"""Pre-launch split/checkpoint/model audit. Does not train or read test labels."""
import hashlib
import json
from pathlib import Path
import torch

PROJECT=Path('/data1/ztx/MyModel-MDTA')
SPLITS=PROJECT/'data/splits/davis_drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final'
BASE=PROJECT/'outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/baseline'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    import sys;sys.path.insert(0,str(PROJECT))
    from datasets.davis_dataset_p13d import DavisDatasetP13D
    from model_periodic_ot import MyModelMDTAP13DPeriodicOT
    ds=DavisDatasetP13D(pairs_csv=PROJECT/'data/raw/davis/pairs.csv',
       drug_1d_dir=PROJECT/'data/processed/davis/drug_1d_chemberta2',
       drug_3d_dir=PROJECT/'data/processed/davis/drug_3d',use_drug_3d=True,
       protein_1d_dir=PROJECT/'data/processed/davis/protein_1d_esm2',
       protein_3d_dir=PROJECT/'data/processed/davis/protein_3d_gvp')
    report=dict(split_root=str(SPLITS),folds=[])
    for fold in range(1,6):
        sp=SPLITS/f'fold_{fold}'/'split.json';s=json.loads(sp.read_text());sets={};audit={}
        for part in ['train','val','test']:
            f=ds.df.iloc[s[part+'_indices']];sets[part]=set(f.drug_id)
            audit[part]=dict(drugs=len(sets[part]),pairs=len(f),proteins=int(f.protein_id.nunique()))
        overlaps={f'{a}_{b}':len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
        bp=BASE/f'fold_{fold}'/'best_model.pt';ck=torch.load(bp,map_location='cpu',weights_only=False)
        if Path(ck['args']['split_json']).resolve()!=sp.resolve():raise ValueError('checkpoint split mismatch')
        report['folds'].append(dict(fold=fold,split_path=str(sp),split_sha256=sha(sp),audit=audit,
          drug_overlap=overlaps,baseline_checkpoint=str(bp),baseline_checkpoint_sha256=sha(bp),
          baseline_best_epoch=ck['epoch'],baseline_val_metrics=ck['val_metrics']))
    torch.manual_seed(42);scratch=MyModelMDTAP13DPeriodicOT()
    report['experiment_b_scratch_parameter_sha256']=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in scratch.parameters())).hexdigest()
    report['experiment_b_checkpoint_load_calls']=0
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
