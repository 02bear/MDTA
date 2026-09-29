"""Frozen, validation-selected five-checkpoint ensemble test for Protected E2 v2."""
import argparse, gc, hashlib, importlib, json, os, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

P=Path('/data1/ztx/MyModel-MDTA')
sys.path.insert(0,str(P))
import train_p13d_earlystop as metric_module
M=importlib.import_module('train_p13d_finegrained_residual_protected_v2_earlystop')
B=P/'outputs/Refine_experiment/davis/cold_start/drug_cold_5fold_seed42'
OUT=B/'evaluation/missing_tests_20260908/protected_ensemble'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def metrics(y,p):return metric_module.compute_regression_metrics(torch.tensor(p,dtype=torch.float32),torch.tensor(y,dtype=torch.float32))
def move(b,device):return M.move_batch_to_device(b,device)

@torch.no_grad()
def infer(model,loader,cfg,tag,device):
 ys=[];ps=[]
 for i,b in enumerate(loader):
  b=move(b,device);out=model(b,return_debug=True,disable_local=False)
  ys.append(b['label'].detach().float().reshape(-1).cpu());ps.append(out['pred'].detach().float().reshape(-1).cpu())
  if i%100==0:print(json.dumps({'tag':tag,'batch':i,'batches':len(loader)}),flush=True)
 return torch.cat(ys).numpy(),torch.cat(ps).numpy()

def run(fold):
 folder=B/f'protected_e2_v2/fold_{fold}'
 selection=json.loads((folder/'checkpoint_selection.json').read_text())
 entries=selection['ensemble_checkpoints'][:5]
 assert len(entries)==5
 paths=[Path(e['path']) if Path(e['path']).is_absolute() else P/e['path'] for e in entries]
 assert len({int(e['epoch']) for e in entries})==5
 cks=[torch.load(x,map_location='cpu',weights_only=False) for x in paths]
 cfg=SimpleNamespace(**cks[0]['args']);metric_module.set_seed(cfg.seed)
 original=json.loads(Path(cfg.split_json).read_text())
 fullpath=P/f'data/splits/davis_drug_cold_5fold_seed42/fold_{fold}/split.json';split=json.loads(fullpath.read_text())
 for k in ('train_indices','val_indices'):assert original[k]==split[k]
 cfg.split_json=str(fullpath)
 ds=M.build_dataset(cfg);frame=ds.df
 vl=M.make_loader(ds,split['val_indices'],cfg,False);tl=M.make_loader(ds,split['test_indices'],cfg,False)
 vys=[];vps=[];tys=[];tps=[];valdiff=[]
 dev=torch.device('cuda')
 for entry,path,ck in zip(entries,paths,cks):
  assert int(ck['epoch'])==int(entry['epoch'])
  model=M.build_model(SimpleNamespace(**ck['args']),dev);model.load_state_dict(ck['model_state_dict'],strict=True);model.eval()
  vy,vp=infer(model,vl,cfg,f'fold{fold}/epoch{ck["epoch"]}/val',dev)
  vm=metrics(vy,vp);ref=ck['val_metrics']['final'];diff={k:abs(vm[k]-ref[k]) for k in vm};assert max(diff.values())<5e-4,diff
  ty,tp=infer(model,tl,cfg,f'fold{fold}/epoch{ck["epoch"]}/TEST',dev)
  vys.append(vy);vps.append(vp);tys.append(ty);tps.append(tp);valdiff.append(diff)
  del model;gc.collect();torch.cuda.empty_cache()
 for x in vys[1:]:assert np.array_equal(x,vys[0])
 for x in tys[1:]:assert np.array_equal(x,tys[0])
 vp=np.mean(vps,axis=0);tp=np.mean(tps,axis=0);vy=vys[0];ty=tys[0]
 tf=frame.iloc[split['test_indices']];assert np.allclose(tf.label,ty)
 pdrows=[]
 for drug in sorted(set(tf.drug_id)):
  mask=tf.drug_id.to_numpy()==drug;pdrows.append({'drug_id':drug,'n':int(mask.sum()),**metrics(ty[mask],tp[mask])})
 result={'method':'protected_ensemble','fold':fold,'epochs':[int(e['epoch']) for e in entries],'checkpoints':[str(x) for x in paths],'checkpoint_sha256':[sha(x) for x in paths],'split':str(fullpath),'split_sha256':sha(fullpath),'validation_individual_reproduction_abs_diff':valdiff,'ensemble_validation_metrics':metrics(vy,vp),'test_metrics':metrics(ty,tp),'per_drug':pdrows,'test_pairs':len(ty),'test_drugs':len(pdrows),'selection_source':'checkpoint_selection.json ensemble_checkpoints first five; validation-selected before test'}
 target=OUT/f'fold_{fold}';target.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(target/'test_predictions.npz',indices=np.asarray(split['test_indices']),y_true=ty,y_pred=tp,drug_ids=tf.drug_id.to_numpy(),protein_ids=tf.protein_id.to_numpy())
 dump(target/'test_metrics.json',result);print('RESULT='+json.dumps(result['test_metrics']),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');ap.add_argument('--folds',nargs='+',type=int);a=ap.parse_args()
 if a.worker:
  torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.12)
  for f in a.folds:run(f)
 else:
  OUT.mkdir(parents=True,exist_ok=True);assign=[(a.folds,'0')] if a.folds else [([1,5],'1'),([2],'0'),([3],'3'),([4],'2')];workers=[]
  for folds,gpu in assign:
   base=f'worker_{"_".join(map(str,folds))}';log=OUT/f'{base}_rerun_{int(time.time())}.log';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
   with log.open('x') as h:p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--folds',*map(str,folds)],cwd=P,env=env,stdout=h,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
   workers.append({'folds':folds,'gpu':gpu,'pid':p.pid})
  dump(OUT/'workers.json',workers);print(json.dumps(workers))
