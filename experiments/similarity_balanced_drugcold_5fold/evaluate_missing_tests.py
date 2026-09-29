"""Frozen checkpoint testing of three completed historical five-fold strategies."""
import argparse, gc, hashlib, importlib, json, os, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

PROJECT=Path('/data1/ztx/MyModel-MDTA')
sys.path.insert(0,str(PROJECT))
import train_p13d_earlystop as metric_module
BASE=PROJECT/'outputs/Refine_experiment/davis/cold_start/drug_cold_5fold_seed42'
OUT=BASE/'evaluation/missing_tests_20260908'
METHODS={
 'strictfp':('e2_guided_strictfp_stageA','train_p13d_e2_guided_strictfp_stageA_cachealigned','best_model.pt'),
 'threegrain':('e2_guided_three_grain_v1_stageA','train_p13d_e2_guided_three_granularity_v1_stageA_frozen_cachealigned','best_model.pt'),
 'protected_overall':('protected_e2_v2','train_p13d_finegrained_residual_protected_v2_earlystop','best_overall_model.pt'),
 'protected_robust':('protected_e2_v2','train_p13d_finegrained_residual_protected_v2_earlystop','best_robust_model.pt')}
WORKER_LABEL=sys.argv[sys.argv.index('--method')+1] if '--method' in sys.argv else 'all'
SELECTED=[WORKER_LABEL] if WORKER_LABEL!='all' else list(METHODS)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj): Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def metrics(y,p): return metric_module.compute_regression_metrics(torch.tensor(p,dtype=torch.float32),torch.tensor(y,dtype=torch.float32))

def progress(**kw):
 dump(OUT/f'status_{WORKER_LABEL}.json',dict(pid=os.getpid(),timestamp=time.time(),**kw))
 print(json.dumps(kw),flush=True)

def make_model(mod,cfg,ck,protected,device):
 if protected: m=mod.build_model(cfg,device)
 else:
  m=mod.E2GuidedThreeGranularityDTA(drug_1d_in_dim=cfg.drug_1d_in_dim,drug_3d_node_in_dim=cfg.drug_3d_node_in_dim,protein_1d_in_dim=1280,protein_3d_node_s_dim=6,protein_3d_node_v_dim=3,hidden_dim=cfg.hidden_dim,dropout=cfg.dropout,task='regression',pocket_top_k=cfg.pocket_top_k,interaction_heads=cfg.interaction_heads,freeze_e2=True).to(device)
 m.load_state_dict(ck['model_state_dict'],strict=True)
 m.eval()
 return m

@torch.no_grad()
def infer(m,loader,mod,cfg,protected,tag,device):
 ps,ys=[],[]
 m.eval()
 for i,b in enumerate(loader):
  b=(mod.move_batch_to_device if protected else mod.move_to_device)(b,device)
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16 if getattr(cfg,'amp_dtype','bf16')=='bf16' else torch.float16,enabled=bool(getattr(cfg,'amp',False))):
   output=m(b,return_debug=True,disable_local=False) if protected else m(b,return_details=True)
  ps.append(output['pred'].detach().float().reshape(-1).cpu())
  ys.append(b['label'].detach().float().reshape(-1).cpu())
  if i%50==0: progress(state='evaluating',tag=tag,batch=i,batches=len(loader))
 return torch.cat(ys).numpy(),torch.cat(ps).numpy()

def run_one(name,f):
 folder,module,filename=METHODS[name]
 protected=name.startswith('protected')
 cp=BASE/f'{folder}/fold_{f}/{filename}'
 before=sha(cp)
 ck=torch.load(cp,map_location='cpu',weights_only=False)
 cfg=SimpleNamespace(**ck['args'])
 metric_module.set_seed(cfg.seed)
 original_split=json.loads(Path(cfg.split_json).read_text())
 full_path=PROJECT/f'data/splits/davis_drug_cold_5fold_seed42/fold_{f}/split.json'
 split=json.loads(full_path.read_text())
 for k in ('train_indices','val_indices'): assert original_split[k]==split[k],(name,f,k)
 cfg.split_json=str(full_path)
 mod=importlib.import_module(module)
 dataset=mod.build_dataset(cfg) if protected else mod.build_datasets(cfg)
 frame=dataset.df if protected else dataset.e2_dataset.df
 raw=pd.read_csv(cfg.pairs_csv,dtype={'drug_id':str,'protein_id':str})
 assert raw[['drug_id','protein_id']].equals(frame[['drug_id','protein_id']])
 assert np.allclose(raw.label,frame.label)
 sets=[set(frame.iloc[split[k]].drug_id) for k in ('train_indices','val_indices','test_indices')]
 assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])
 if protected:
  vl=mod.make_loader(dataset,split['val_indices'],cfg,False)
  tl=mod.make_loader(dataset,split['test_indices'],cfg,False)
 else: _,vl,tl=mod.build_loaders(cfg,dataset,split)
 dev=torch.device('cuda')
 model=make_model(mod,cfg,ck,protected,dev)
 vy,vp=infer(model,vl,mod,cfg,protected,f'{name}/fold{f}/validation',dev)
 vm=metrics(vy,vp)
 ref=ck['val_metrics']['final'] if protected else ck['val_result']['final']
 delta={k:abs(vm[k]-ref[k]) for k in vm}
 assert max(delta.values())<1e-4,dict(validation_reproduction_failed=delta)
 ty,tp=infer(model,tl,mod,cfg,protected,f'{name}/fold{f}/TEST',dev)
 tf=frame.iloc[split['test_indices']]
 assert np.allclose(tf.label,ty)
 tm=metrics(ty,tp)
 target=OUT/name/f'fold_{f}';target.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(target/'test_predictions.npz',y_true=ty,y_pred=tp,drug_ids=tf.drug_id.to_numpy(),protein_ids=tf.protein_id.to_numpy(),indices=np.array(split['test_indices']))
 pdrows=[]
 for drug in sorted(set(tf.drug_id)):
  mask=tf.drug_id.to_numpy()==drug
  pdrows.append(dict(drug_id=drug,n=int(mask.sum()),**metrics(ty[mask],tp[mask])))
 assert before==sha(cp)
 result=dict(method=name,fold=f,epoch=ck['epoch'],checkpoint=str(cp),checkpoint_sha256=before,split=str(full_path),split_sha256=sha(full_path),batch_size=cfg.batch_size,validation_metrics=vm,validation_reproduction_abs_diff=delta,test_metrics=tm,per_drug=pdrows,drug_disjoint=True,test_pairs=len(ty),test_drugs=len(sets[2]))
 dump(target/'test_metrics.json',result)
 print('RESULT='+json.dumps(result['test_metrics']),flush=True)
 del model,ck,vl,tl,dataset
 gc.collect();torch.cuda.empty_cache()
 return result

def worker():
 torch.set_num_threads(4)
 # Resource cap changes allocator budget only; input batches and precision unchanged.
 torch.cuda.set_per_process_memory_fraction(0.12)
 results=[];errors=[]
 for name in SELECTED:
  for f in range(1,6):
   try:
    saved=OUT/name/f'fold_{f}/test_metrics.json'
    if saved.exists():
     prior=json.loads(saved.read_text())
     assert prior['checkpoint_sha256']==sha(prior['checkpoint'])
     results.append(prior)
    else: results.append(run_one(name,f))
   except Exception as e:
    import traceback
    traceback.print_exc()
    errors.append(dict(method=name,fold=f,error=repr(e)))
    dump(OUT/f'errors_{WORKER_LABEL}.json',errors)
    gc.collect();torch.cuda.empty_cache()
   dump(OUT/f'partial_results_{WORKER_LABEL}.json',results)
 summary={}
 for name in SELECTED:
  rr=[r for r in results if r['method']==name]
  if len(rr)==5:
   zs=[np.load(OUT/name/f'fold_{f}/test_predictions.npz',allow_pickle=True) for f in range(1,6)]
   indices=np.concatenate([z['indices'] for z in zs]);assert sorted(indices)==list(range(30056))
   summary[name]={'macro':{k:float(np.mean([r['test_metrics'][k] for r in rr])) for k in rr[0]['test_metrics']},'sample_sd':{k:float(np.std([r['test_metrics'][k] for r in rr],ddof=1)) for k in rr[0]['test_metrics']},'pooled':metrics(np.concatenate([z['y_true'] for z in zs]),np.concatenate([z['y_pred'] for z in zs]))}
 dump(OUT/f'SUMMARY_{WORKER_LABEL}.json',dict(results=results,errors=errors,summary=summary,protocol='frozen existing checkpoints; protected overall and robust reported separately; no selection on test; ensembles not evaluated'))
 progress(state='completed' if not errors else 'completed_with_errors',successful_runs=len(results),errors=errors,summary=summary)

def launch():
 OUT.mkdir(parents=True,exist_ok=False)
 cps={}
 for name,(folder,mod,filename) in METHODS.items():
  for f in range(1,6):
   p=BASE/f'{folder}/fold_{f}/{filename}';cps[str(p)]=sha(p)
 dump(OUT/'checkpoint_manifest.json',cps)
 with (OUT/'evaluate.log').open('x') as log:
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
  proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 print(json.dumps(dict(pid=proc.pid,output=str(OUT))),flush=True)

if __name__=='__main__':
 worker() if '--worker' in sys.argv else launch()
