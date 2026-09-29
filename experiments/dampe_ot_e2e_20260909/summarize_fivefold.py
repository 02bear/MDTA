import argparse
import json
from pathlib import Path
import numpy as np

PROJECT=Path('/data1/ztx/MyModel-MDTA')
ROOT=PROJECT/'outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final'

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);a=p.parse_args()
    root=ROOT/a.experiment;rows=[]
    for fold in range(1,6):
        path=root/f'fold_{fold}'/'metrics.json'
        if not path.exists():raise FileNotFoundError(path)
        value=json.loads(path.read_text())
        if not value.get('complete') or value.get('smoke'):raise ValueError('Invalid formal result '+str(path))
        rows.append(value)
    stats={}
    for part in ['val','test']:
        stats[part]={}
        for metric in ['mse','ci','rm2']:
            x=np.array([r[f'{part}_{metric}'] for r in rows],float)
            stats[part][metric]=dict(mean=float(x.mean()),sample_std=float(x.std(ddof=1)))
    output=dict(experiment=a.experiment,folds=rows,mean_and_sample_std=stats,ddof=1,complete=True)
    (root/'fivefold_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=[a.experiment,'Fold  BestEpoch  VAL_MSE  VAL_CI  VAL_RM2  TEST_MSE  TEST_CI  TEST_RM2']
    for r in rows:lines.append(f'{r["fold"]}  {r["best_epoch"]}  {r["val_mse"]:.6f}  {r["val_ci"]:.6f}  {r["val_rm2"]:.6f}  {r["test_mse"]:.6f}  {r["test_ci"]:.6f}  {r["test_rm2"]:.6f}')
    for part in ['val','test']:
        lines.append(part.upper()+': '+', '.join(f'{m}={stats[part][m]["mean"]:.6f} ± {stats[part][m]["sample_std"]:.6f}' for m in ['mse','ci','rm2']))
    (root/'fivefold_summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':main()
