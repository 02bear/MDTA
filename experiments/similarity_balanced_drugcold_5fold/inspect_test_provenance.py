import json,sys,hashlib
from pathlib import Path
import torch,numpy as np,pandas as pd
P=Path('/data1/ztx/MyModel-MDTA');B=P/'outputs/Refine_experiment/davis/cold_start/drug_cold_5fold_seed42'
for f in range(1,6):
 c=torch.load(P/f'experiments/pdbbind_to_davis_transfer/data/global_predictions/fold{f}.pt',map_location='cpu',weights_only=False)
 cp=B/f'baseline/fold_{f}/best_model.pt';ck=torch.load(cp,map_location='cpu',weights_only=False)
 h=json.loads((B/f'evaluation/baseline_e2_5fold/fold_{f}/test_metrics.json').read_text())
 split=json.loads((P/f'data/splits/davis_drug_cold_5fold_seed42/fold_{f}/split.json').read_text())
 out={'fold':f,'checkpoint_epoch':ck.get('epoch'),'checkpoint_keys':list(ck),'cache_keys':list(c),'cache_meta':{k:v for k,v in c.items() if k in ('checkpoint','checkpoint_sha256','args','metadata')},'historical':h['baseline'],'cache_test_mse':float(torch.mean((c['label'][split['test_indices']]-c['prediction'][split['test_indices']])**2)),'current_ck_sha':hashlib.sha256(cp.read_bytes()).hexdigest()}
 print(json.dumps(out,default=str),flush=True)
