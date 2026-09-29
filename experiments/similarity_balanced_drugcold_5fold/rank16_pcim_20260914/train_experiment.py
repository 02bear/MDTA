"""Five-fold, validation-selected frozen Rank16 + PCIM + locked R1 experiment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
from pair_interaction import PairResidual
from common import HERE, OUT, PROJECT, FeatureStore, R1, dump, save, sha, seed_all, rng_state, restore_rng, metric, verify_manifest, load_protocol


def pairs(drugs, n_proteins):
    return np.repeat(drugs,n_proteins), np.tile(np.arange(n_proteins),len(drugs))


@torch.no_grad()
def infer(model, store, query, n_proteins, batch_size, perturb=None):
    model.eval()
    d,p = pairs(query,n_proteins)
    output=[]
    for start in range(0,len(d),batch_size):
        b = store.batch(d[start:start+batch_size],p[start:start+batch_size],perturb,
                        global_only=model.variant=='global_mlp')
        output.append(model(b).cpu().numpy())
    return np.concatenate(output).reshape(len(query),n_proteins).astype(np.float64)


def select_scale(base, deltas, r1, val_drugs, val_labels, lambdas):
    choices=[]
    for scale in lambdas:
        prediction = base + scale*deltas
        corrected,_,_ = r1.correct(prediction,val_drugs)
        error=corrected-val_labels
        choices.append({'lambda':float(scale),'mse':float(np.mean(error**2)),
                        'mae':float(np.mean(np.abs(error)))})
    # Smallest lambda wins exact ties, including the baseline boundary.
    return min(choices,key=lambda x:(x['mse'],x['lambda'])),choices


def train_run(fold,variant,seed):
    verify_manifest()
    cfg=load_protocol()
    dest=OUT/f'runs/{variant}/seed_{seed}/fold_{fold}'
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():
        print('RUN_REUSED',fold,variant,seed,flush=True)
        return
    cache=torch.load(OUT/f'cache/fold_{fold}.pt',map_location='cpu',weights_only=False)
    audit=json.loads((OUT/f'cache/fold_{fold}_audit.json').read_text())
    assert sha(OUT/f'cache/fold_{fold}.pt')==audit['cache_sha256']
    seed_all(seed)
    store=FeatureStore(cache)
    model=PairResidual(variant,top_k=cfg['top_k'],max_delta=cfg['max_delta']).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    train,val,test=(cache['split_drugs'][part] for part in ['train','val','test'])
    n_proteins=len(cache['proteins'])
    r1=R1(cache)
    base=cache['base']
    # Training/selection cannot access test labels through this array.
    y_development=cache['labels'].copy()
    y_development[test]=np.nan
    base_val=r1.correct(base,val)[0]
    baseline_mse=float(np.mean((base_val-y_development[val])**2))
    best={'mse':baseline_mse,'lambda':0.0,'epoch':0}
    best_active=None
    history=[]
    stale=0
    start_epoch=1
    if (dest/'latest.pt').exists():
        ck=torch.load(dest/'latest.pt',map_location='cpu',weights_only=False)
        assert ck['protocol']==cfg and ck['variant']==variant and ck['seed']==seed and ck['fold']==fold
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        best=ck['best'];best_active=ck['best_active'];history=ck['history'];stale=ck['stale']
        start_epoch=ck['epoch']+1
        restore_rng(ck['rng'])
        print('RESUME',fold,variant,seed,start_epoch,flush=True)
    else:
        save(dest/'best.pt',{'model':model.state_dict(),'selection':best.copy(),'protocol':cfg,
                             'variant':variant,'seed':seed,'fold':fold})
    td,tp=pairs(train,n_proteins)
    started=time.time()
    print('TRAIN_START',json.dumps(dict(fold=fold,variant=variant,seed=seed,
          parameters=sum(p.numel() for p in model.parameters()),baseline_validation_mse=baseline_mse)),flush=True)
    for epoch in range(start_epoch,cfg['epochs']+1):
        if stale>=cfg['patience']: break
        began=time.time()
        model.train()
        order=np.random.permutation(len(td))
        losses=[];raw_mse=[];norms=[]
        for step,start in enumerate(range(0,len(order),cfg['batch_size'])):
            ix=order[start:start+cfg['batch_size']]
            d,p=td[ix],tp[ix]
            b=store.batch(d,p,global_only=variant=='global_mlp')
            target=torch.as_tensor(y_development[d,p]-base[d,p],dtype=torch.float32,device='cuda')
            assert torch.isfinite(target).all()
            opt.zero_grad(set_to_none=True)
            delta=model(b)
            mse=(delta-target).square().mean()
            loss=mse+cfg['residual_l2']*delta.square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
            opt.step()
            losses.append(float(loss.detach()));raw_mse.append(float(mse.detach()));norms.append(float(norm))
            if step%200==0:
                print(f'TRAIN fold={fold} variant={variant} seed={seed} epoch={epoch} step={step}/{len(td)//cfg["batch_size"]} mse={raw_mse[-1]:.6f}',flush=True)
        train_delta=infer(model,store,train,n_proteins,cfg['eval_batch_size'])
        val_delta=infer(model,store,val,n_proteins,cfg['eval_batch_size'])
        all_delta=np.zeros_like(base);all_delta[train]=train_delta;all_delta[val]=val_delta
        chosen,choices=select_scale(base,all_delta,r1,val,y_development[val],cfg['lambdas'])
        improved=chosen['mse'] < best['mse']-cfg['min_delta']
        if improved:
            best={**chosen,'epoch':epoch};stale=0
            save(dest/'best.pt',{'model':model.state_dict(),'selection':best.copy(),'protocol':cfg,
                                 'variant':variant,'seed':seed,'fold':fold})
            np.savez_compressed(dest/'best_development_predictions.npz',train_delta=train_delta,val_delta=val_delta,
                                train_drugs=train,val_drugs=val,lambda_value=best['lambda'])
        else: stale+=1
        active=min([x for x in choices if x['lambda']>0],key=lambda x:(x['mse'],x['lambda']))
        if best_active is None or active['mse']<best_active['mse']:
            best_active={**active,'epoch':epoch}
            save(dest/'best_active.pt',{'model':model.state_dict(),'selection':best_active.copy(),
                                      'protocol':cfg,'variant':variant,'seed':seed,'fold':fold})
        record=dict(epoch=epoch,train_loss=float(np.mean(losses)),train_mse=float(np.mean(raw_mse)),
                    max_gradient_norm=max(norms),validation_candidates=choices,best=best.copy(),stale=stale,
                    train_delta_rms=float(np.sqrt(np.mean(train_delta**2))),
                    val_delta_rms=float(np.sqrt(np.mean(val_delta**2))),seconds=time.time()-began)
        history.append(record)
        save(dest/'latest.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,
             best_active=best_active,history=history,stale=stale,rng=rng_state(),protocol=cfg,variant=variant,seed=seed,fold=fold))
        dump(dest/'history.json',history)
        dump(dest/'progress.json',dict(state='training',fold=fold,variant=variant,seed=seed,**record,updated=time.time()))
        print('EPOCH',json.dumps(record),flush=True)
    dump(dest/'selection_locked.json',dict(best=best,best_active=best_active,baseline_validation_mse=baseline_mse,
          selection_data='validation only; test labels not accessed',finished=time.time(),training_seconds=time.time()-started))
    # Test starts only after selection is persisted; never choose variants/seeds on test metrics.
    evaluate_run(dest,cache,store,model,cfg)


def evaluate_run(dest,cache,store,model,cfg):
    selected=torch.load(dest/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model']);model.eval()
    scale=selected['selection']['lambda']
    n_proteins=len(cache['proteins'])
    all_drugs=list(range(len(cache['drugs'])))
    delta=infer(model,store,all_drugs,n_proteins,cfg['eval_batch_size'])
    prediction=cache['base']+scale*delta
    test=cache['split_drugs']['test'];train=cache['split_drugs']['train'];val=cache['split_drugs']['val']
    r1=R1(cache)
    base_r1,_,_=r1.correct(cache['base'],test)
    corrected,correction,alpha=r1.correct(prediction,test)
    if scale==0: assert np.array_equal(corrected,base_r1)
    poisoned=dict(cache);poisoned['labels']=cache['labels'].copy();poisoned['labels'][val+test]=np.nan
    assert np.array_equal(R1(poisoned).correct(prediction,test)[0],corrected)
    d,p=pairs(test,n_proteins)
    table=pd.DataFrame({'pair_index':cache['pair_indices'][test].ravel(),
                        'drug_id':[cache['drugs'][x] for x in d],'protein_id':[cache['proteins'][x] for x in p],
                        'label':cache['labels'][test].ravel(),'rank16':cache['base'][test].ravel(),
                        'rank16_R1':base_r1.ravel(),'delta_raw':delta[test].ravel(),
                        'lambda':scale,'rank16_pcim':prediction[test].ravel(),'rank16_pcim_R1':corrected.ravel(),
                        'R1_correction':correction.ravel(),'R1_alpha':alpha.ravel()})
    table.to_csv(dest/'test_predictions.csv',index=False)
    tests={name:metric(table[name].to_numpy(),table.label.to_numpy()) for name in
           ['rank16','rank16_R1','rank16_pcim','rank16_pcim_R1']}
    # Deterministic validation diagnostics; no test feedback is used for model selection.
    diagnostics={}
    if model.variant!='global_mlp':
        clean=infer(model,store,val,n_proteins,cfg['eval_batch_size'])
        for perturb in ['mismatch_drug_local','shuffle_aa']:
            altered=infer(model,store,val,n_proteins,cfg['eval_batch_size'],perturb)
            diagnostics[perturb]={'raw_delta_change_rms':float(np.sqrt(np.mean((altered-clean)**2))),
                                  'raw_delta_abs_change':float(np.mean(np.abs(altered-clean)))}
        vd,vp=pairs(val,n_proteins)
        b=store.batch(vd[:16],vp[:16])
        with torch.no_grad(): _,info=model(b,diagnostics=True)
        diagnostics['pair_sample']={'mean_entropy':float(info['entropy'].mean()),
            'distinct_atoms':[int(torch.unique(x[m]).numel()) for x,m in zip(info['atom_index'],info['valid'])],
            'distinct_residues':[int(torch.unique(x[m]).numel()) for x,m in zip(info['residue_index'],info['valid'])]}
        np.savez_compressed(dest/'validation_pair_attributions.npz',drug_indices=vd[:16],protein_indices=vp[:16],
                            **{k:v.cpu().numpy() for k,v in info.items()})
    by_drug=[]
    for drug,group in table.groupby('drug_id'):
        b=float(np.mean((group.rank16_R1-group.label)**2))
        n=float(np.mean((group.rank16_pcim_R1-group.label)**2))
        by_drug.append({'drug_id':drug,'base_mse':b,'new_mse':n,'new_minus_base':n-b})
    result=dict(fold=cache['fold'],variant=model.variant,seed=selected['seed'],selection=selected['selection'],
                parameters=sum(p.numel() for p in model.parameters()),test=tests,per_drug=by_drug,
                diagnostics=diagnostics,label_poisoning_invariance=True,baseline_fallback=scale==0,
                r1_config=cache['r1_config'],r1_residual_source='training labels minus combined Rank16+local prediction',
                backbone_checkpoint=cache['checkpoint'],backbone_sha256=cache['checkpoint_sha256'],
                adapter_sha256=sha(dest/'best.pt'),finished=time.time())
    dump(dest/'result.json',result)
    dump(dest/'progress.json',dict(state='completed',fold=cache['fold'],variant=model.variant,
                                 seed=selected['seed'],selection=selected['selection'],updated=time.time()))
    print('TEST_RESULT',json.dumps(result),flush=True)


def aggregate():
    cfg=load_protocol()
    entries=[json.loads(p.read_text()) for p in sorted((OUT/'runs').glob('*/seed_*/fold_*/result.json'))]
    grouped={}
    for variant in cfg['variants']:
        grouped[variant]={}
        for seed in cfg['seeds']:
            selected=[x for x in entries if x['variant']==variant and x['seed']==seed]
            if not selected:continue
            item={'folds_completed':[x['fold'] for x in selected],'enabled_folds':[x['fold'] for x in selected if not x['baseline_fallback']],
                  'complete':len(selected)==5}
            if len(selected)==5:
                item['macro']={name:{key:float(np.mean([r['test'][name][key] for r in selected])) for key in
                                    ['mse','rmse','mae','ci','rm2']} for name in ['rank16_R1','rank16_pcim_R1']}
                difference=np.array([d['new_minus_base'] for r in selected for d in r['per_drug']])
                assert len(difference)==68
                generator=np.random.default_rng(20260914)
                boot=difference[generator.integers(0,68,size=(10000,68))].mean(1)
                item['drug_macro_bootstrap']={'mean':float(difference.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
                                              'improved':int((difference<0).sum()),'worsened':int((difference>0).sum()),
                                              'unchanged':int((difference==0).sum())}
            grouped[variant][str(seed)]=item
    report={'completed_runs':len(entries),'planned_runs':len(cfg['variants'])*len(cfg['seeds'])*5,
            'complete':len(entries)==len(cfg['variants'])*len(cfg['seeds'])*5,'groups':grouped,
            'warning':'Existing repeatedly inspected split; development evidence. Seeds vary adapters only, backbone seed42 fixed.',
            'updated':time.time()}
    dump(OUT/'summary.json',report)
    lines=['# Rank16 + conditional pair interaction + R1','',
           f'Completed {report["completed_runs"]}/{report["planned_runs"]} runs.','',
           '| Variant | Seed | Fold | Lambda | Baseline+R1 MSE | New+R1 MSE |','|---|---:|---:|---:|---:|---:|']
    for r in entries:
        lines.append(f'| {r["variant"]} | {r["seed"]} | {r["fold"]} | {r["selection"]["lambda"]} | {r["test"]["rank16_R1"]["mse"]:.6f} | {r["test"]["rank16_pcim_R1"]["mse"]:.6f} |')
    lines+=['',report['warning']]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report


def worker():
    cfg=load_protocol()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'logs').mkdir(exist_ok=True)
    lock=(OUT/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Every fold of the primary model is run before the controls.
    tasks=[('cache',f,None,None) for f in range(1,6)]
    tasks += [('train',f,v,s) for v in cfg['variants'] for s in cfg['seeds'] for f in range(1,6)]
    try:
        for position,(action,fold,variant,seed) in enumerate(tasks):
            verify_manifest()
            if action=='cache' and (OUT/f'cache/fold_{fold}_audit.json').exists():continue
            if action=='train' and (OUT/f'runs/{variant}/seed_{seed}/fold_{fold}/result.json').exists():continue
            while True:
                free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free>=24000:break
                dump(OUT/'status.json',dict(state='waiting_for_gpu',gpu=0,free_mib=free,pid=os.getpid(),updated=time.time()))
                time.sleep(30)
            if action=='cache':
                command=[sys.executable,'-B','-u',str(HERE/'build_cache.py'),'--fold',str(fold)]
                name=f'cache_fold{fold}'
            else:
                command=[sys.executable,'-B','-u',str(HERE/'train_experiment.py'),'train','--fold',str(fold),'--variant',variant,'--seed',str(seed)]
                name=f'{variant}_seed{seed}_fold{fold}'
            logpath=OUT/f'logs/{name}.log'
            with logpath.open('a') as log:
                process=subprocess.Popen(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                status=dict(state='running',action=action,fold=fold,variant=variant,seed=seed,gpu=0,
                            pid=os.getpid(),child_pid=process.pid,task_index=position+1,total_tasks=len(tasks),
                            log=str(logpath),updated=time.time())
                dump(OUT/'status.json',status)
                code=process.wait()
            if code:raise RuntimeError(f'{name} exited {code}; see {logpath}')
            if action=='train':aggregate()
        aggregate()
        dump(OUT/'status.json',dict(state='completed',gpu=0,pid=os.getpid(),finished=time.time()))
    except Exception as exc:
        dump(OUT/'status.json',dict(state='failed',error=str(exc),pid=os.getpid(),traceback=traceback.format_exc(),updated=time.time()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','worker','aggregate'])
    parser.add_argument('--fold',type=int,choices=range(1,6))
    parser.add_argument('--variant',choices=['pair_graph','pair_pool','global_mlp'])
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.action=='train':train_run(args.fold,args.variant,args.seed)
    elif args.action=='worker':worker()
    else:print(json.dumps(aggregate(),indent=2))
