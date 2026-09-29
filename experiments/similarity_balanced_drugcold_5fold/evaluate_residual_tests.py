"""Add held-out tests using locked R1/R2/N3 configurations and residual banks."""
import json, sys, os, subprocess, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
P=Path('/data1/ztx/MyModel-MDTA')
B=P/'outputs/Refine_experiment/davis/cold_start/drug_cold_5fold_seed42'
OUT=B/'evaluation/missing_tests_20260908/residual'
sys.path[:0]=[str(P),str(P/'experiments/klifs85_interaction'),str(P/'experiments/r1_crossfit_robustness')]
from experiments.klifs85_interaction import run_residual_kernel as r1
from experiments.klifs85_interaction.train_klifs_interact import metrics
from n3_common import R1_CONFIGS, SEED

def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def checked(y,p,ref):
 m=metrics(y.ravel(),p.ravel())
 diff={k:abs(m[k]-ref[k]) for k in ('mse','ci','rm2')}
 assert max(diff.values())<1e-4,diff
 return diff

def main():
 torch.set_num_threads(2)
 sp=P/'experiments/klifs85_interaction/data/similarity_audit_fold1/entity_similarities.npz'
 sim=np.load(sp,allow_pickle=True)
 ds=[str(x) for x in sim['drug_ids']];ps=[str(x) for x in sim['protein_ids']]
 dl={x:i for i,x in enumerate(ds)};pl={x:i for i,x in enumerate(ps)}
 D=sim['drug_similarity'].astype(float);S=sim['protein_similarity'].astype(float)
 results=[];frames=[];allshuffle=[]
 for f in range(1,6):
  print(f'FOLD {f} start',flush=True)
  splitpath=P/f'data/splits/davis_drug_cold_5fold_seed42/fold_{f}/split.json'
  split=json.loads(splitpath.read_text())
  tr,va,te=[np.array([dl[str(x)] for x in split[k]]) for k in ('train_drugs','val_drugs','test_drugs')]
  assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
  cachepath=P/f'experiments/pdbbind_to_davis_transfer/data/global_predictions/fold{f}.pt'
  cache=torch.load(cachepath,map_location='cpu',weights_only=False)
  ri=np.full((len(ds),len(ps)),-1,dtype=int)
  for i,(d,p) in enumerate(zip(cache['drug_id'],cache['protein_id'])): ri[dl[str(d)],pl[str(p)]]=i
  assert (ri>=0).all()
  for ids,k in zip((tr,va,te),('train_indices','val_indices','test_indices')):
   assert np.array_equal(np.sort(ri[ids].ravel()),np.sort(split[k]))
  y=cache['label'].numpy()[ri].astype(float);base=cache['prediction'].numpy()[ri].astype(float)
  # Residual arrays are zero outside the original outer-training drug set.
  rin=np.zeros_like(base);rin[tr]=y[tr]-base[tr]
  roof=np.zeros_like(base)
  n3dir=P/f'experiments/r1_crossfit_robustness/outputs/fold_{f}'
  oofpath=n3dir/'oof_predictions.csv'
  oof=pd.read_csv(oofpath,dtype={'drug_id':str,'protein_id':str})
  assert not oof.duplicated(['drug_id','protein_id']).any()
  assert set(oof.drug_id)=={ds[d] for d in tr} and len(oof)==len(tr)*442
  assert (oof.groupby('drug_id').size()==442).all()
  for row in oof.itertuples(): roof[dl[row.drug_id],pl[row.protein_id]]=row.residual_oof
  rpath=P/f'experiments/klifs85_interaction/outputs/residual_kernel_fold{f}_v2/results.json'
  saved=json.loads(rpath.read_text());cfg=R1_CONFIGS[f]
  assert all(saved['inner']['R1_best'][k]==v for k,v in cfg.items())
  r2s={}
  for name,key in [('R2','R2_best'),('R2_active','R2_best_active')]:
   c=dict(saved['inner'][key]);c['smoother'],c['supported']=r1.protein_smoother(S,c['eta'],c['k_protein'],c['min_protein_similarity']);r2s[name]=c
  def predict(q):
   z={'P13D_cache':base[q]}
   z['R1_in']=r1.outer_predictions(q,tr,rin,base,D,cfg)[0]
   z['R1_OOF']=r1.outer_predictions(q,tr,roof,base,D,cfg)[0]
   z['TargetMean_OOF']=base[q]+roof[tr].mean(0)[None,:]
   for name,c in r2s.items():z[name]=r1.outer_predictions(q,tr,rin,base,D,cfg,c)[0]
   return z
  # Reproduction of all original validation conditions before test scoring.
  vv=predict(va);n3ref=json.loads((n3dir/'metrics.json').read_text())['metrics']
  diffs={}
  for name in ('P13D_cache','R1_in','R1_OOF','TargetMean_OOF'):
   diffs[name]=checked(y[va],vv[name],n3ref['P13D' if name=='P13D_cache' else name])
  for name in r2s:diffs[name]=checked(y[va],vv[name],saved['outer_validation'][name])
  histpath=B/f'evaluation/baseline_e2_5fold/fold_{f}/test_metrics.json'
  hist=json.loads(histpath.read_text())
  histpred=pd.read_csv(histpath.with_name('test_predictions.csv'),dtype={'drug_id':str,'protein_id':str})
  assert set(histpred.pair_index)==set(split['test_indices'])
  matched=np.array([base[dl[row.drug_id],pl[row.protein_id]] for row in histpred.itertuples()])
  cache_diff=float(np.max(np.abs(matched-histpred.baseline_pred.to_numpy())))
  native_metrics=hist['baseline']['test_global_metrics'];cache_metrics=metrics(y[te].ravel(),base[te].ravel())
  cache_metric_diff={k:abs(cache_metrics[k]-native_metrics[k]) for k in ('mse','ci','rm2')}
  zz=predict(te)
  fm={n:metrics(y[te].ravel(),p.ravel()) for n,p in zz.items()}
  frame=pd.DataFrame({'fold':f,'pair_index':ri[te].ravel(),'drug_id':np.repeat([ds[d] for d in te],442),'protein_id':np.tile(ps,len(te)),'label':y[te].ravel(),**{n:p.ravel() for n,p in zz.items()}})
  frames.append(frame)
  rng=np.random.default_rng(SEED);shuffle_preds=[];sm=[]
  for j in range(100):
   sr=np.zeros_like(roof);sr[tr]=roof[tr][rng.permutation(len(tr))]
   pred=r1.outer_predictions(te,tr,sr,base,D,cfg)[0]
   shuffle_preds.append(pred.ravel())
   # Controls were locked at 100 whole-drug permutations; no best control selection.
   sm.append(float(np.mean((pred-y[te])**2)))
  allshuffle.append(np.stack(shuffle_preds))
  rr={'fold':f,'metrics':fm,'r1_config':cfg,'validation_reproduction_abs_diff':diffs,'cache_vs_historical_native_test_max_abs':cache_diff,'cache_vs_historical_native_test_metric_abs_diff':cache_metric_diff,'split_sha256':sha(splitpath),'global_cache_sha256':sha(cachepath),'oof_file_sha256':sha(oofpath),'r1_config_file_sha256':sha(rpath),'test_drugs':len(te),'test_pairs':len(te)*442,'shuffle_MSE':sm,'shuffle_unit':'entire 442-dimensional training-drug residual row','seed':SEED,'reference_drugs':'original outer train only','test_labels_used_for_correction':False}
  dump(OUT/f'fold_{f}/metrics.json',rr);frame.to_csv(OUT/f'fold_{f}/test_predictions.csv',index=False)
  results.append(rr);print(json.dumps({'fold':f,'test_MSE':{k:v['mse'] for k,v in fm.items()}}),flush=True)
 allframe=pd.concat(frames,ignore_index=True)
 assert sorted(allframe.pair_index)==list(range(30056))
 summary={}
 for name in results[0]['metrics']:
  summary[name]={'macro':{k:float(np.mean([r['metrics'][name][k] for r in results])) for k in ('mse','ci','rm2')},'sample_sd':{k:float(np.std([r['metrics'][name][k] for r in results],ddof=1)) for k in ('mse','ci','rm2')},'pooled':metrics(allframe.label.to_numpy(),allframe[name].to_numpy())}
  allframe[name+'_sqerror']=(allframe[name]-allframe.label)**2
 perdrug=allframe.groupby(['fold','drug_id'])[[n+'_sqerror' for n in summary]].mean().reset_index()
 boot={}
 for name in ('R1_in','R1_OOF','TargetMean_OOF','R2','R2_active'):
  delta=(perdrug[name+'_sqerror']-perdrug.P13D_cache_sqerror).to_numpy()
  rng=np.random.default_rng(42);means=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
  boot[name]={'mean_delta_mse':float(delta.mean()),'95_ci':np.quantile(means,[.025,.975]).tolist(),'improved_drugs':int((delta<0).sum()),'drugs':len(delta),'cluster':'test drug, all 442 pairs; conditional on fixed trained models'}
 spred=np.concatenate(allshuffle,axis=1);smse=np.mean((spred-allframe.label.to_numpy()[None,:])**2,axis=1)
 allframe.to_csv(OUT/'test_predictions_allfolds.csv',index=False);perdrug.to_csv(OUT/'per_drug.csv',index=False)
 dump(OUT/'SUMMARY.json',{'results':results,'summary':summary,'bootstrap_vs_P13D':boot,'shuffled_R1_OOF_pooled_MSE':smse.tolist(),'protocol':'post-hoc heldout-test supplement using original factorized P13D caches; all original hyperparameters and 100 drug-row permutations fixed; no retraining; cache predictions are not claimed prediction-equivalent to native batched P13D; historical primary N3 verdict unchanged'})
 print('COMPLETED',flush=True)

if __name__=='__main__':
 if '--worker' in sys.argv: main()
 else:
  OUT.mkdir(parents=True,exist_ok=True)
  with (OUT/'evaluate_resume.log').open('x') as log:
   p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],cwd=P,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  print(p.pid)
