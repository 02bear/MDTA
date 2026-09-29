"""Complete the locked rank16 trial: train 2/4/5, evaluate all folds, fixed R1."""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROJECT = Path('/data1/ztx/MyModel-MDTA')
ROOT = PROJECT/'outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final'
OLD = ROOT/'bilinear_rank16_trial_20260907'
OUT = ROOT/'bilinear_rank16_complete_20260914'
sys.path.insert(0, str(HERE/'source'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def check_sources():
    manifest = json.loads((HERE/'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        assert sha(name)==digest, f'Locked source/data changed: {name}'
    return manifest


def modules():
    import model_impl as h
    return h, h.base


def model_folder(fold):
    return (OLD if fold in (1,3) else OUT)/f'fold_{fold}'


def atomic_save(path, obj):
    import torch
    path=Path(path)
    for attempt in range(3):
        temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
        try:
            with temp.open('wb') as f:
                torch.save(obj, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp,path)
            return
        except OSError as exc:
            print(f'SAVE_RETRY path={path} attempt={attempt+1} error={exc!r}',flush=True)
            if temp.exists():
                temp.unlink()
            if attempt==2:
                raise
            time.sleep(2*(attempt+1))


def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def train(fold):
    assert fold in (2,4,5), 'Never retrain completed folds 1/3'
    check_sources()
    import numpy as np
    import torch
    h,t=modules()
    folder=model_folder(fold)
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training_complete.json').exists():
        print('REUSE completed training',fold,flush=True)
        return
    cfg=json.loads((HERE/f'configs/fold_{fold}.json').read_text())
    args=SimpleNamespace(**cfg)
    t.set_seed(args.seed)
    assert torch.cuda.is_available()
    device=torch.device('cuda')
    dataset,train_set,val_set,train_loader,val_loader=t.build_dataloaders(args)
    split=json.loads(Path(args.split_json).read_text())
    assert list(train_set.indices)==split['train_indices']
    assert list(val_set.indices)==split['val_indices']
    t.save_split_indices(train_set,val_set,folder)
    model=h.build_model(args,device)
    assert all(p.requires_grad for p in model.parameters())
    criterion=torch.nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    history=[]
    best_rmse=float('inf'); best_epoch=-1; counter=0; best_train=None; best_val=None
    start=1
    latest=folder/'latest_model.pt'
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        assert ck['args']==vars(args), 'Resume config changed'
        model.load_state_dict(ck['model_state_dict'],strict=True)
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        # load_state_dict normally moves optimizer tensors to the parameter device.
        history=ck['history']; best_rmse=ck['best_val_rmse']; best_epoch=ck['best_epoch']
        counter=ck['epochs_no_improve']; best_train=ck['best_train_metrics']; best_val=ck['best_val_metrics']
        start=ck['epoch']+1
        restore_rng(ck['rng'])
        del ck
        print(f'RESUME fold={fold} next_epoch={start}',flush=True)
    print(f'TRAIN_START fold={fold} train={len(train_set)} val={len(val_set)} seed={args.seed} batch={args.batch_size} trainable={sum(p.numel() for p in model.parameters())}',flush=True)
    for epoch in range(start,args.epochs+1):
        if args.early_stop_patience>0 and counter>=args.early_stop_patience:
            break
        began=time.time()
        tm=t.train_one_epoch(model,train_loader,criterion,optimizer,device,log_interval=200)
        vm=h.evaluate(model,val_loader,criterion,device)
        assert np.isfinite([tm['loss'],vm['mse'],vm['rmse']]).all(), 'Nonfinite loss'
        assert all(torch.isfinite(p).all().item() for p in model.parameters()), 'Nonfinite parameters'
        history.append(dict(epoch=epoch,train=tm,val=vm))
        improved=vm['rmse']<best_rmse-args.early_stop_min_delta
        if improved:
            best_rmse=vm['rmse']; best_epoch=epoch; counter=0
            best_train=dict(tm); best_val=dict(vm)
        else:
            counter+=1
        ck=dict(epoch=epoch,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                train_metrics=tm,val_metrics=vm,args=vars(args),history=history,
                best_val_rmse=best_rmse,best_epoch=best_epoch,epochs_no_improve=counter,
                best_train_metrics=best_train,best_val_metrics=best_val,rng=rng_state())
        if improved:
            atomic_save(folder/'best_model.pt',ck)
            np.savez_compressed(folder/'best_val_predictions.npz',**h._last_validation)
        atomic_save(latest,ck)
        dump(folder/'history.json',history)
        dump(folder/'progress.json',dict(fold=fold,epoch=epoch,best_epoch=best_epoch,best_val_mse=best_val['mse'],
                                        val_mse=vm['mse'],counter=counter,seconds=time.time()-began,pid=os.getpid(),updated=time.time()))
        print(f'EPOCH={epoch} TRAIN_MSE={tm["mse"]:.8f} VAL_MSE={vm["mse"]:.8f} BEST_EPOCH={best_epoch} COUNTER={counter} SECONDS={time.time()-began:.1f}',flush=True)
    assert best_epoch>0
    dump(folder/'best_summary.json',dict(best_epoch=best_epoch,best_train_metrics=best_train,best_val_metrics=best_val))
    dump(folder/'training_complete.json',dict(fold=fold,best_epoch=best_epoch,last_epoch=history[-1]['epoch'],checkpoint_sha256=sha(folder/'best_model.pt'),finished=time.time()))
    print(f'TRAIN_FINISHED fold={fold} best_epoch={best_epoch}',flush=True)


def fixed_r1(pred, label, df, train_ids, test_ids, config):
    import numpy as np
    # Execute only the original three pure kernel functions; no selector/evaluator.
    source=(HERE/'r1_source.py').read_text()
    tree=ast.parse(source)
    names={'top_weights','drug_kernel_stats','apply_correction'}
    scope={'np':np}
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<locked-r1>','exec'),scope)
    sim=np.load(HERE/'entity_similarities.npz',allow_pickle=True)
    drug_ids=list(map(str,sim['drug_ids'].tolist()))
    protein_ids=list(map(str,sim['protein_ids'].tolist()))
    di={d:i for i,d in enumerate(drug_ids)}
    lookup={(str(d),str(p)):i for i,(d,p) in enumerate(zip(df.drug_id,df.protein_id))}
    train_rows=np.array([[lookup[(d,p)] for p in protein_ids] for d in train_ids])
    test_rows=np.array([[lookup[(d,p)] for p in protein_ids] for d in test_ids])
    # Construct only training residuals. No query label is accessed here.
    residual=np.asarray(label[train_rows]-pred[train_rows],dtype=np.float64)
    mean,var,support=scope['drug_kernel_stats'](np.array([di[d] for d in test_ids]),np.array([di[d] for d in train_ids]),residual,sim['drug_similarity'].astype(np.float64),config)
    corrected,correction,alpha=scope['apply_correction'](pred[test_rows],mean,var,support,config)
    return test_rows,corrected,correction,alpha


def evaluate(fold):
    check_sources()
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    h,t=modules()
    folder=OUT/f'evaluation/fold_{fold}'
    folder.mkdir(parents=True,exist_ok=True)
    ckpath=model_folder(fold)/'best_model.pt'
    digest=sha(ckpath)
    if (folder/'results.json').exists():
        saved=json.loads((folder/'results.json').read_text())
        assert saved['checkpoint_sha256']==digest
        assert (folder/'test_predictions.csv').exists()
        print('REUSE evaluation',fold,flush=True)
        return
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    args=SimpleNamespace(**ck['args'])
    assert args.head_rank==16
    split=json.loads(Path(args.split_json).read_text())
    assert Path(args.split_json).resolve()==(PROJECT/f'data/splits/davis_drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/fold_{fold}/split.json').resolve()
    train_ids=list(map(str,split['train_drugs'])); val_ids=list(map(str,split['val_drugs'])); test_ids=list(map(str,split['test_drugs']))
    assert set(train_ids).isdisjoint(val_ids) and set(train_ids).isdisjoint(test_ids) and set(val_ids).isdisjoint(test_ids)
    t.set_seed(args.seed)
    device=torch.device('cuda')
    dataset,tr,va,_,val_loader=t.build_dataloaders(args)
    df=dataset.df.copy()
    raw=pd.read_csv(args.pairs_csv,dtype={'drug_id':str,'protein_id':str})
    assert len(df)==30056 and df[['drug_id','protein_id']].equals(raw[['drug_id','protein_id']])
    assert np.allclose(df.label,raw.label)
    df['drug_id']=df.drug_id.astype(str); df['protein_id']=df.protein_id.astype(str)
    assert set(df.iloc[tr.indices].drug_id)==set(train_ids)
    assert set(df.iloc[va.indices].drug_id)==set(val_ids)
    model=h.build_model(args,device)
    model.load_state_dict(ck['model_state_dict'],strict=True)
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    reproduced=h.evaluate(model,val_loader,torch.nn.MSELoss(),device)
    diffs={k:abs(reproduced[k]-ck['val_metrics'][k]) for k in ['mse','rmse','mae','ci','rm2']}
    assert max(diffs.values())<1e-4,diffs
    oldpred=np.load(model_folder(fold)/'best_val_predictions.npz',allow_pickle=True)
    pred_diff=float(np.max(np.abs(oldpred['y_pred']-h._last_validation['y_pred'])))
    assert pred_diff<1e-4,pred_diff
    dump(folder/'validation_replay.json',dict(fold=fold,checkpoint_sha256=digest,metric_abs_diff=diffs,pred_max_abs_diff=pred_diff,passed=True))
    print(f'VALIDATION_REPLAY_PASSED fold={fold} diffs={diffs}',flush=True)
    preds=[]; ys=[]
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=t.mdta_collate_fn_p13d,pin_memory=True)
    model.eval()
    with torch.inference_mode():
        offset=0
        for step,batch in enumerate(loader):
            n=len(batch['drug_id'])
            assert list(map(str,batch['drug_id']))==df.drug_id.iloc[offset:offset+n].tolist()
            assert list(map(str,batch['protein_id']))==df.protein_id.iloc[offset:offset+n].tolist()
            offset+=n
            batch=t.move_batch_to_device(batch,device)
            preds.append(model(batch).reshape(-1).cpu())
            ys.append(batch['label'].reshape(-1).cpu())
            if step%200==0:
                print(f'INFER fold={fold} step={step}/{len(loader)}',flush=True)
    pred=torch.cat(preds).numpy().astype(np.float64)
    y=torch.cat(ys).numpy().astype(np.float64)
    assert np.isfinite(pred).all() and np.allclose(y,df.label,atol=1e-6)
    refs=json.loads((HERE/'r1_reference_summary.json').read_text())
    ref=next(x for x in refs['folds'] if x['fold']==fold)
    config={k:ref['R1_best'][k] for k in ['gamma','k_drug','min_drug_similarity','tau','beta','clip','scale']}
    test_rows,corrected,correction,alpha=fixed_r1(pred,y,df,train_ids,test_ids,config)
    # Meaningful invariance check: poisoned validation/test labels cannot affect correction.
    poisoned=y.copy(); poisoned[~df.drug_id.isin(train_ids).to_numpy()]=np.nan
    _,corrected2,_,_=fixed_r1(pred,poisoned,df,train_ids,test_ids,config)
    assert np.array_equal(corrected,corrected2)
    ii=test_rows.ravel()
    assert sorted(ii.tolist())==sorted(split['test_indices'])
    output=df.iloc[ii][['drug_id','protein_id']].reset_index(drop=True)
    output.insert(0,'pair_index',ii); output.insert(0,'fold',fold)
    output['label']=y[ii]; output['rank16']=pred[ii]; output['rank16_R1']=corrected.ravel()
    output['R1_correction']=correction.ravel(); output['R1_alpha']=alpha.ravel()
    def metric(a,b):
        return t.compute_regression_metrics(torch.tensor(a,dtype=torch.float64),torch.tensor(b,dtype=torch.float64))
    m={name:metric(output[name].to_numpy(),output.label.to_numpy()) for name in ['rank16','rank16_R1']}
    temp=folder/'test_predictions.csv.tmp'; output.to_csv(temp,index=False); os.replace(temp,folder/'test_predictions.csv')
    train_indices=np.array(tr.indices)
    np.savez_compressed(folder/'train_predictions.npz',pair_index=train_indices,prediction=pred[train_indices],label=y[train_indices])
    per=[]
    for drug,g in output.groupby('drug_id'):
        item={'drug_id':drug,'n_pairs':len(g)}
        for name in ['rank16','rank16_R1']:
            err=g[name].to_numpy()-g.label.to_numpy()
            item[name]=dict(**metric(g[name].to_numpy(),g.label.to_numpy()),bias=float(err.mean()),centered_mse=float(np.mean((err-err.mean())**2)))
        per.append(item)
    result=dict(fold=fold,checkpoint=str(ckpath),checkpoint_sha256=digest,split=args.split_json,split_sha256=sha(args.split_json),
                best_epoch=ck['epoch'],test=m,baseline_test=ref['test'],r1_config=config,validation_reproduction=diffs,
                test_rows=len(output),test_drugs=len(test_ids),label_poisoning_invariance=True,
                r1_reference='this fold training drugs; residuals from this rank16 checkpoint',r1_selection='locked historical per-fold settings; no retuning',per_drug=per,finished=time.time())
    dump(folder/'results.json',result)
    print('TEST_RESULT='+json.dumps({k:result[k] for k in ['fold','best_epoch','test','baseline_test']}),flush=True)


def aggregate():
    import numpy as np
    import pandas as pd
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'aggregate.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        results=[json.loads(p.read_text()) for p in sorted((OUT/'evaluation').glob('fold_*/results.json'))]
        rows=[]
        for r in results:
            row={'fold':r['fold']}
            for name,metric in [('baseline',r['baseline_test']['P13D']),('baseline_R1',r['baseline_test']['R1']),*r['test'].items()]:
                for key in ['mse','rmse','mae','ci','rm2']:
                    row[name+'_'+key]=metric[key]
            rows.append(row)
        summary={'completed_folds':[r['fold'] for r in results],'complete':len(results)==5,'per_fold':rows}
        if len(results)==5:
            frames=[pd.read_csv(OUT/f'evaluation/fold_{f}/test_predictions.csv',dtype={'drug_id':str,'protein_id':str}) for f in range(1,6)]
            allpred=pd.concat(frames,ignore_index=True)
            assert len(allpred)==30056 and allpred.pair_index.nunique()==30056 and allpred.drug_id.nunique()==68
            assert allpred.groupby('drug_id').fold.nunique().max()==1
            summary['macro']={key:{'mean':float(np.mean([row[key] for row in rows])),'sample_std':float(np.std([row[key] for row in rows],ddof=1))} for key in rows[0] if key!='fold'}
            temp=OUT/'test_predictions_allfolds.csv.tmp'; allpred.to_csv(temp,index=False); os.replace(temp,OUT/'test_predictions_allfolds.csv')
        dump(OUT/'summary.json',summary)
        lines=['# Rank16 new split: test comparison', '',f'Completed folds: {summary["completed_folds"]}. Complete: {summary["complete"]}.', '', '| Fold | Baseline MSE | Baseline+R1 | Rank16 | Rank16+R1 |','|---|---:|---:|---:|---:|']
        for row in rows:
            lines.append(f'| {row["fold"]} | {row["baseline_mse"]:.6f} | {row["baseline_R1_mse"]:.6f} | {row["rank16_mse"]:.6f} | {row["rank16_R1_mse"]:.6f} |')
        if summary['complete']:
            lines.append('| Mean | '+' | '.join(f'{summary["macro"][k+"_mse"]["mean"]:.6f}' for k in ['baseline','baseline_R1','rank16','rank16_R1'])+' |')
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def worker(gpu):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'logs').mkdir(exist_ok=True)
    lock=(OUT/f'worker_gpu{gpu}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=([('evaluate',1),('train',2),('evaluate',2),('train',5),('evaluate',5)] if gpu==0 else [('evaluate',3),('train',4),('evaluate',4)])
    status=OUT/f'worker_gpu{gpu}.json'
    for action,fold in tasks:
        check_sources()
        while True:
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=34000:
                break
            dump(status,dict(state='waiting_for_gpu',gpu=gpu,action=action,fold=fold,free_mib=free,pid=os.getpid(),updated=time.time()))
            time.sleep(30)
        env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        logfile=OUT/f'logs/{action}_fold{fold}.log'
        with logfile.open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),action,'--fold',str(fold)],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
            dump(status,dict(state='running',gpu=gpu,action=action,fold=fold,pid=os.getpid(),child_pid=proc.pid,log=str(logfile),updated=time.time()))
            print(f'START gpu={gpu} action={action} fold={fold} pid={proc.pid}',flush=True)
            code=proc.wait()
        if code:
            dump(status,dict(state='failed',gpu=gpu,action=action,fold=fold,exit_code=code,pid=os.getpid(),log=str(logfile),updated=time.time()))
            raise RuntimeError(f'{action} fold{fold} exit={code}; inspect log, checkpoint allows explicit resume')
        if action=='evaluate':
            aggregate()
    dump(status,dict(state='completed',gpu=gpu,pid=os.getpid(),finished=time.time()))
    print(f'WORKER_FINISHED gpu={gpu}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','evaluate','worker','aggregate'])
    parser.add_argument('--fold',type=int,choices=range(1,6))
    parser.add_argument('--gpu',type=int,choices=[0,1])
    args=parser.parse_args()
    if args.action=='train': train(args.fold)
    elif args.action=='evaluate': evaluate(args.fold)
    elif args.action=='worker': worker(args.gpu)
    else: aggregate()
