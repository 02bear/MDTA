import json, sys
from pathlib import Path
import numpy as np
import torch
P=Path('/data1/ztx/MyModel-MDTA');sys.path.insert(0,str(P))
import train_p13d_earlystop as metric_module
O=P/'outputs/Refine_experiment/davis/cold_start/drug_cold_5fold_seed42/evaluation/missing_tests_20260908/protected_ensemble'
rows=[];zs=[]
for f in range(1,6):
 rows.append(json.loads((O/f'fold_{f}/test_metrics.json').read_text()))
 zs.append(np.load(O/f'fold_{f}/test_predictions.npz',allow_pickle=True))
indices=np.concatenate([z['indices'] for z in zs]);assert sorted(indices)==list(range(30056))
keys=('mse','rmse','mae','ci','rm2')
summary={
 'macro':{k:float(np.mean([r['test_metrics'][k] for r in rows])) for k in keys},
 'sample_sd':{k:float(np.std([r['test_metrics'][k] for r in rows],ddof=1)) for k in keys},
 'pooled':metric_module.compute_regression_metrics(torch.tensor(np.concatenate([z['y_pred'] for z in zs])),torch.tensor(np.concatenate([z['y_true'] for z in zs]))),
 'folds':rows,
 'audit':{'all_five_folds_present':True,'test_indices_cover_0_through_30055_exactly_once':True,'selection':'first five ensemble_checkpoints in each pre-existing checkpoint_selection.json; no test selection','validation_numeric_tolerance':5e-4,'maximum_observed_validation_metric_abs_difference':max(max(d.values()) for r in rows for d in r['validation_individual_reproduction_abs_diff'])}
}
(O/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:summary[k] for k in ('macro','sample_sd','pooled','audit')},indent=2))
