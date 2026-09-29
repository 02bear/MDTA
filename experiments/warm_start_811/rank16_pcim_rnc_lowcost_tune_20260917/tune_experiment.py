"""Low-cost, leakage-audited tuning for the completed DAVIS warm-start run."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = Path('/data1/ztx/MyModel-MDTA')
SOURCE_OUT = PROJECT/'outputs/Refine_experiment/davis/warm_start/random_pair_811_seed42/rank16_pcim_pair_rnc_20260917'
OUT = PROJECT/'outputs/Refine_experiment/davis/warm_start/random_pair_811_seed42/rank16_pcim_rnc_lowcost_tune_20260917'
sys.path.insert(0, str(HERE/'source'))
sys.path.insert(0, str(HERE))

from pair_interaction import PairResidual
from rnc_loss import StratifiedPairSampler, rank_n_contrast_loss
import train_p13d_earlystop as base_module


def protocol():
    return json.loads((HERE/'tune_protocol.json').read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}


def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])


class FeatureStore:
    def __init__(self,cache,device='cuda'):
        from torch.nn.utils.rnn import pad_sequence
        self.device=device;self.atom_length=[len(x) for x in cache['atoms']];self.protein_length=[len(x) for x in cache['residues']]
        self.atoms=pad_sequence(cache['atoms'],batch_first=True).to(device);self.residues=pad_sequence(cache['residues'],batch_first=True).to(device)
        self.aa=pad_sequence(cache['aa'],batch_first=True,padding_value=20).to(device)
        self.drug_global=cache['drug_global'].to(device);self.protein_global=cache['protein_global'].to(device)
        self.atom_mask=torch.arange(self.atoms.size(1),device=device)[None]<torch.tensor(self.atom_length,device=device)[:,None]
        self.residue_mask=torch.arange(self.residues.size(1),device=device)[None]<torch.tensor(self.protein_length,device=device)[:,None]

    def batch(self,drug,protein):
        d=torch.as_tensor(drug,device=self.device);p=torch.as_tensor(protein,device=self.device)
        nd=max(self.atom_length[int(x)] for x in drug);nr=max(self.protein_length[int(x)] for x in protein)
        return {'drug_global':self.drug_global[d],'protein_global':self.protein_global[p],
                'atoms':self.atoms[d,:nd],'atom_mask':self.atom_mask[d,:nd],
                'residues':self.residues[p,:nr],'residue_mask':self.residue_mask[p,:nr],'aa':self.aa[p,:nr]}


def load_cache():
    cache_path=SOURCE_OUT/'cache/warm_811.pt';audit=json.loads((SOURCE_OUT/'cache/audit.json').read_text())
    assert sha(cache_path)==audit['cache_sha256']
    cache=torch.load(cache_path,map_location='cpu',weights_only=False)
    assert cache['checkpoint_sha256']==protocol()['fixed_backbone']['checkpoint_sha256']
    return cache


def development_labels(cache):
    labels=cache['labels'].copy();test=cache['split_pairs']['test'];labels[test['drug'],test['protein']]=np.nan
    return labels


def masked_stats(qd,qp,residual,mask,similarity,cfg):
    mean=np.zeros(len(qd));var=np.zeros(len(qd));support=np.zeros(len(qd))
    for i,(d,p) in enumerate(zip(qd,qp)):
        refs=np.flatnonzero(mask[:,p]);sims=similarity[d,refs]
        order=np.argsort(sims)[::-1][:min(cfg['k_drug'],len(refs))];refs=refs[order];sims=sims[order]
        keep=sims>=cfg['min_drug_similarity'];refs=refs[keep];sims=sims[keep]
        if not len(refs): continue
        w=sims**cfg['gamma'];total=w.sum()
        if total<=1e-12: continue
        values=residual[refs,p];mean[i]=np.sum(w*values)/total;var[i]=np.sum(w*(values-mean[i])**2)/total
        support[i]=total/cfg['k_drug']
    return mean,var,support


def apply_r1(base,mean,var,support,cfg):
    alpha=cfg['scale']*support/(support+cfg['tau'])*np.exp(-cfg['beta']*var)
    correction=alpha*np.clip(mean,-cfg['clip'],cfg['clip'])
    return base+correction,correction,alpha


def r1_correct(prediction,labels,train,query,similarity,cfg):
    mask=np.zeros_like(prediction,dtype=bool);mask[train['drug'],train['protein']]=True
    residual=np.zeros_like(prediction);residual[train['drug'],train['protein']]=labels[train['drug'],train['protein']]-prediction[train['drug'],train['protein']]
    mean,var,support=masked_stats(query['drug'],query['protein'],residual,mask,similarity,cfg)
    return apply_r1(prediction[query['drug'],query['protein']],mean,var,support,cfg)


@torch.no_grad()
def predict_pairs(model,store,pairs,batch_size=64):
    model.eval();out=[]
    for start in range(0,len(pairs['drug']),batch_size):
        out.append(model(store.batch(pairs['drug'][start:start+batch_size],pairs['protein'][start:start+batch_size])).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def metric(pred,label):
    return base_module.compute_regression_metrics(torch.tensor(pred,dtype=torch.float64),torch.tensor(label,dtype=torch.float64))


def current_config():
    ck=torch.load(SOURCE_OUT/'adapter/best.pt',map_location='cpu',weights_only=False);p=ck['protocol'];a=p['adapter'];r=p['rnc']
    return {'variant':'bidirectional','width':a['width'],'top_k':a['top_k'],'coverage_budget':a['coverage_budget'],
            'max_delta':a['max_delta'],'cross_max_scale':a['cross_max_scale'],'residual_l2':a['residual_l2'],
            'rnc_weight':r['weight'],'temperature':r['temperature'],'projection_dim':r['projection_dim'],
            'rnc_batch_size':r['batch_size'],'microbatch_size':r['microbatch_size'],'interval':r['interval'],
            'high_count':r['high_count'],'mid_count':r['mid_count'],'high_threshold':r['high_threshold'],
            'mid_threshold':r['mid_threshold'],'warmup_epochs':r['warmup_epochs'],'seed':p['seed']}


def make_model(cfg):
    return PairResidual(cfg['variant'],width=cfg['width'],top_k=cfg['top_k'],max_delta=cfg['max_delta'],
        selection='coverage',coverage_budget=cfg['coverage_budget'],cross_max_scale=cfg['cross_max_scale'],
        enable_contrast=True,contrast_dim=cfg['projection_dim']).cuda()


def load_model(cfg,checkpoint):
    model=make_model(cfg);ck=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True);return model


def expanded_calibration(cache,train_delta,val_delta,label):
    cfg=protocol()['calibration'];train=cache['split_pairs']['train'];val=cache['split_pairs']['val'];dev=development_labels(cache)
    mask=np.zeros_like(cache['base'],dtype=bool);mask[train['drug'],train['protein']]=True
    target=dev[val['drug'],val['protein']];coarse=[]
    for lam in cfg['lambda']:
        residual=np.zeros_like(cache['base']);residual[train['drug'],train['protein']]=dev[train['drug'],train['protein']]-(cache['base'][train['drug'],train['protein']]+lam*train_delta)
        base=cache['base'][val['drug'],val['protein']]+lam*val_delta
        for gamma in cfg['gamma']:
          for k in cfg['k_drug']:
           for minimum in cfg['min_drug_similarity']:
            kernel={'gamma':gamma,'k_drug':k,'min_drug_similarity':minimum}
            mean,var,support=masked_stats(val['drug'],val['protein'],residual,mask,cache['similarity'],kernel)
            for tau in cfg['tau']:
             for beta in cfg['beta']:
              item={**kernel,'tau':tau,'beta':beta,'clip':1.0,'scale':1.0}
              pred,_,_=apply_r1(base,mean,var,support,item);mse=float(np.mean((pred-target)**2))
              coarse.append({'lambda':float(lam),**item,'mse':mse})
    coarse.sort(key=lambda x:x['mse']);fine=[]
    for seed_cfg in coarse[:cfg['coarse_keep']]:
        lam=seed_cfg['lambda'];kernel={k:seed_cfg[k] for k in ['gamma','k_drug','min_drug_similarity']}
        residual=np.zeros_like(cache['base']);residual[train['drug'],train['protein']]=dev[train['drug'],train['protein']]-(cache['base'][train['drug'],train['protein']]+lam*train_delta)
        mean,var,support=masked_stats(val['drug'],val['protein'],residual,mask,cache['similarity'],kernel)
        base=cache['base'][val['drug'],val['protein']]+lam*val_delta
        for clip in cfg['clip']:
         for scale in cfg['scale']:
          item={**kernel,'tau':seed_cfg['tau'],'beta':seed_cfg['beta'],'clip':clip,'scale':scale}
          pred,_,_=apply_r1(base,mean,var,support,item);mse=float(np.mean((pred-target)**2))
          fine.append({'lambda':float(lam),**item,'mse':mse})
    fine.sort(key=lambda x:x['mse']);best=fine[0]
    result={'label':label,'best':best,'coarse_top':coarse[:cfg['coarse_keep']],'fine_top':fine[:30],
            'current_validation_mse':protocol()['current_validation_mse'],'gain':protocol()['current_validation_mse']-best['mse']}
    return result


def candidate_dir(name): return OUT/'candidates'/name


def train_candidate(name,cfg,store,cache):
    dest=candidate_dir(name);dest.mkdir(parents=True,exist_ok=True);result_path=dest/'result.json'
    if result_path.exists(): return json.loads(result_path.read_text())
    scfg=protocol()['screening'];dev=development_labels(cache);train=cache['split_pairs']['train'];val=cache['split_pairs']['val']
    labels_train=dev[train['drug'],train['protein']];seed_all(cfg['seed']);model=make_model(cfg)
    opt=torch.optim.AdamW(model.parameters(),lr=scfg['lr'],weight_decay=scfg['weight_decay'])
    sampler=StratifiedPairSampler(labels_train,batch_size=cfg['rnc_batch_size'],high_count=cfg['high_count'],mid_count=cfg['mid_count'],
        high_threshold=cfg['high_threshold'],mid_threshold=cfg['mid_threshold'])
    start=1;best={'mse':float('inf'),'epoch':0,'lambda':0.0};stale=0;history=[];latest=dest/'latest.pt'
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        start=ck['epoch']+1;best=ck['best'];stale=ck['stale'];history=ck['history'];restore_rng(ck['rng'])
        print('RESUME',name,start,flush=True)
    total_steps=(len(labels_train)+scfg['batch_size']-1)//scfg['batch_size']
    for epoch in range(start,scfg['max_epochs']+1):
        if stale>=scfg['patience']: break
        began=time.time();model.train();order=np.random.permutation(len(labels_train));losses=[];mses=[];rncs=[]
        for step,pos in enumerate(range(0,len(order),scfg['batch_size'])):
            ix=order[pos:pos+scfg['batch_size']];d=train['drug'][ix];p=train['protein'][ix]
            target=torch.as_tensor(dev[d,p]-cache['base'][d,p],dtype=torch.float32,device='cuda')
            opt.zero_grad(set_to_none=True);delta=model(store.batch(d,p));mse=(delta-target).square().mean()
            task=mse+cfg['residual_l2']*delta.square().mean();task.backward();rnc_value=0.0
            if cfg['rnc_weight']>0 and (step+1)%cfg['interval']==0:
                rix=sampler.sample();rd=train['drug'][rix];rp=train['protein'][rix];emb=[]
                for rp0 in range(0,len(rix),cfg['microbatch_size']):
                    _,z=model(store.batch(rd[rp0:rp0+cfg['microbatch_size']],rp[rp0:rp0+cfg['microbatch_size']]),return_contrast=True);emb.append(z)
                z=torch.cat(emb);aff=torch.as_tensor(labels_train[rix],dtype=torch.float32,device='cuda')
                rloss,_=rank_n_contrast_loss(z,aff,temperature=cfg['temperature'],mode='standard',high_threshold=cfg['high_threshold'])
                warm=min(1.0,((epoch-1)+(step+1)/total_steps)/cfg['warmup_epochs']);(cfg['rnc_weight']*cfg['interval']*warm*rloss).backward()
                rnc_value=float(rloss.detach());rncs.append(rnc_value)
            torch.nn.utils.clip_grad_norm_(model.parameters(),scfg['grad_clip'],error_if_nonfinite=True);opt.step()
            losses.append(float(task.detach())+cfg['rnc_weight']*rnc_value);mses.append(float(mse.detach()))
            if step%400==0: print(f'TRAIN {name} epoch={epoch} step={step}/{total_steps} mse={mses[-1]:.6f}',flush=True)
        val_delta=predict_pairs(model,store,val,scfg['eval_batch_size']);target_val=dev[val['drug'],val['protein']]
        choices=[]
        for lam in scfg['screen_lambdas']:
            pred=cache['base'][val['drug'],val['protein']]+lam*val_delta;choices.append({'lambda':lam,'mse':float(np.mean((pred-target_val)**2))})
        chosen=min(choices,key=lambda x:(x['mse'],x['lambda']));improved=chosen['mse']<best['mse']-scfg['min_delta']
        if improved:
            best={**chosen,'epoch':epoch};stale=0;save(dest/'best.pt',{'model':model.state_dict(),'config':cfg,'best':best})
        else: stale+=1
        rec={'epoch':epoch,'train_mse':float(np.mean(mses)),'train_loss':float(np.mean(losses)),
             'rnc_loss':float(np.mean(rncs)) if rncs else 0.0,'validation':choices,'best':best.copy(),'stale':stale,'seconds':time.time()-began}
        history.append(rec);save(latest,{'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best':best,
             'stale':stale,'history':history,'rng':rng_state(),'config':cfg});dump(dest/'progress.json',rec)
        print('EPOCH',name,json.dumps(rec),flush=True)
    assert best['epoch']>0
    result={'name':name,'config':cfg,'best':best,'checkpoint':str(dest/'best.pt'),'epochs':history[-1]['epoch'],'complete':True}
    dump(result_path,result);del model;torch.cuda.empty_cache();return result


def current_deltas(store,cache,parts=('train','val')):
    cfg=current_config();model=load_model(cfg,SOURCE_OUT/'adapter/best.pt')
    result={part:predict_pairs(model,store,cache['split_pairs'][part]) for part in parts};del model;torch.cuda.empty_cache();return cfg,result


def stage_calibrate():
    path=OUT/'stage1_current_calibration.json'
    if path.exists(): print('CALIBRATION_REUSED',flush=True);return
    cache=load_cache();store=FeatureStore(cache);_,d=current_deltas(store,cache)
    result=expanded_calibration(cache,d['train'],d['val'],'current_epoch17');dump(path,result)
    print('CALIBRATION_RESULT',json.dumps(result['best']),flush=True)


def stage_structure():
    aggregate=OUT/'stage2_structure.json'
    if aggregate.exists(): print('STRUCTURE_REUSED',flush=True);return
    cache=load_cache();store=FeatureStore(cache);base=current_config();configs=[]
    for top in protocol()['pcim_search']['top_k']:
      for budget in protocol()['pcim_search']['coverage_budget']:
        configs.append({**base,'top_k':top,'coverage_budget':budget,'width':32,'seed':42})
    for width in [16,64]: configs.append({**base,'top_k':64,'coverage_budget':32,'width':width,'seed':42})
    results=[]
    for cfg in configs:
        name=f"pcim_k{cfg['top_k']}_c{cfg['coverage_budget']}_w{cfg['width']}"
        results.append(train_candidate(name,cfg,store,cache))
    results.sort(key=lambda x:x['best']['mse']);dump(aggregate,{'complete':True,'ranking':results})
    print('STRUCTURE_BEST',json.dumps(results[0]),flush=True)


def stage_rnc():
    aggregate=OUT/'stage3_rnc.json'
    if aggregate.exists(): print('RNC_REUSED',flush=True);return
    structure=json.loads((OUT/'stage2_structure.json').read_text());base=structure['ranking'][0]['config'];cache=load_cache();store=FeatureStore(cache);results=[]
    for weight in protocol()['rnc_search']['weights']:
        cfg={**base,'rnc_weight':weight,'temperature':2.0,'seed':42};name=f"rnc_w{weight:g}_t2"
        results.append(train_candidate(name,cfg,store,cache))
    nonzero=[x for x in results if x['config']['rnc_weight']>0];best_weight=min(nonzero,key=lambda x:x['best']['mse'])['config']['rnc_weight']
    for temperature in [1.0,4.0]:
        cfg={**base,'rnc_weight':best_weight,'temperature':temperature,'seed':42};name=f"rnc_w{best_weight:g}_t{temperature:g}"
        results.append(train_candidate(name,cfg,store,cache))
    results.sort(key=lambda x:x['best']['mse']);dump(aggregate,{'complete':True,'best_screen_weight':best_weight,'ranking':results})
    print('RNC_BEST',json.dumps(results[0]),flush=True)


def load_meta_deltas(meta,store,cache,parts=('train','val')):
    if meta.get('source')=='current': cfg=current_config();checkpoint=SOURCE_OUT/'adapter/best.pt'
    else: cfg=meta['config'];checkpoint=Path(meta['checkpoint'])
    model=load_model(cfg,checkpoint);d={part:predict_pairs(model,store,cache['split_pairs'][part]) for part in parts}
    del model;torch.cuda.empty_cache();return cfg,d


def locked_prediction(cache,train_delta,query_delta,query,calibration,labels):
    train=cache['split_pairs']['train'];pred=cache['base'].copy();lam=calibration['lambda']
    pred[train['drug'],train['protein']]+=lam*train_delta;pred[query['drug'],query['protein']]+=lam*query_delta
    r1={k:calibration[k] for k in ['gamma','k_drug','min_drug_similarity','tau','beta','clip','scale']}
    return r1_correct(pred,labels,train,query,cache['similarity'],r1),pred[query['drug'],query['protein']]


def stage_final():
    result_path=OUT/'result.json'
    if result_path.exists(): print('FINAL_REUSED',flush=True);return
    cache=load_cache();store=FeatureStore(cache);rnc=json.loads((OUT/'stage3_rnc.json').read_text())
    candidates=[{'source':'current','name':'current_epoch17'}]+rnc['ranking'][:3];calibrations=[];delta_cache={}
    for meta in candidates:
        cfg,d=load_meta_deltas(meta,store,cache);label=meta['name'];cal=expanded_calibration(cache,d['train'],d['val'],label)
        calibrations.append({'meta':meta,'config':cfg,'calibration':cal});delta_cache[label]=d
        print('FINAL_CALIBRATION',label,json.dumps(cal['best']),flush=True)
    calibrations.sort(key=lambda x:x['calibration']['best']['mse']);selected=calibrations[0];cfg=selected['config'];seed42_meta=selected['meta']
    confirmation=[{'seed':42,'meta':seed42_meta,'config':cfg}]
    for seed in [2026,3407]:
        ccfg={**cfg,'seed':seed};name=f"confirm_{selected['calibration']['label']}_seed{seed}"
        confirmation.append({'seed':seed,'meta':train_candidate(name,ccfg,store,cache),'config':ccfg})
    locked=selected['calibration']['best'];dev=development_labels(cache);val=cache['split_pairs']['val'];target=dev[val['drug'],val['protein']]
    all_d=[];seed_metrics=[]
    for item in confirmation:
        if item['seed']==42: d=delta_cache[selected['calibration']['label']]
        else: _,d=load_meta_deltas(item['meta'],store,cache)
        pred=locked_prediction(cache,d['train'],d['val'],val,locked,dev)[0][0]
        seed_metrics.append({'seed':item['seed'],'validation_mse':float(np.mean((pred-target)**2))});all_d.append(d)
    ensemble_train=np.mean([d['train'] for d in all_d],axis=0);ensemble_val=np.mean([d['val'] for d in all_d],axis=0)
    ensemble_pred=locked_prediction(cache,ensemble_train,ensemble_val,val,locked,dev)[0][0];ensemble_mse=float(np.mean((ensemble_pred-target)**2))
    single_mse=seed_metrics[0]['validation_mse'];choice='ensemble' if ensemble_mse<single_mse else 'single_seed42'
    selection={'choice':choice,'calibration':locked,'single_validation_mse':single_mse,'ensemble_validation_mse':ensemble_mse,
               'seed_validation':seed_metrics,'selected_config':cfg,'source_candidate':selected['calibration']['label'],
               'selection_data':'validation only; test labels sealed','locked_at':time.time()}
    dump(OUT/'calibration_ranking.json',{'ranking':calibrations});dump(OUT/'selection_locked.json',selection)
    # Test labels are first used below this persisted selection boundary.
    test=cache['split_pairs']['test'];all_test=[];all_train=[]
    for item in confirmation:
        parts=('train','test');_,d=load_meta_deltas(item['meta'],store,cache,parts=parts);all_train.append(d['train']);all_test.append(d['test'])
    if choice=='ensemble': train_delta=np.mean(all_train,axis=0);test_delta=np.mean(all_test,axis=0)
    else: train_delta=all_train[0];test_delta=all_test[0]
    labels=cache['labels'];(corrected,correction,alpha),raw=locked_prediction(cache,train_delta,test_delta,test,locked,labels)
    poisoned=labels.copy();v=cache['split_pairs']['val'];poisoned[v['drug'],v['protein']]=np.nan;poisoned[test['drug'],test['protein']]=np.nan
    corrected2=locked_prediction(cache,train_delta,test_delta,test,locked,poisoned)[0][0];assert np.array_equal(corrected,corrected2)
    y=labels[test['drug'],test['protein']];base=cache['base'][test['drug'],test['protein']]
    current=json.loads((SOURCE_OUT/'adapter/result.json').read_text())
    result={'complete':True,'selection':selection,'test':{'rank16':metric(base,y),'tuned_raw':metric(raw,y),'tuned_R1':metric(corrected,y)},
            'current_reference':current['test'],'high_affinity_count':int((y>=7).sum()),
            'high_affinity':{'tuned_R1':metric(corrected[y>=7],y[y>=7])},'label_poisoning_invariance':True,'finished':time.time()}
    dump(result_path,result);print('FINAL_RESULT',json.dumps(result),flush=True)


def preflight():
    cache=load_cache();assert [len(cache['split_pairs'][x]['index']) for x in ['train','val','test']]==[24044,3005,3007]
    ck=torch.load(SOURCE_OUT/'adapter/best.pt',map_location='cpu',weights_only=False);cfg=current_config();model=make_model(cfg);model.load_state_dict(ck['model'])
    store=FeatureStore(cache);train=cache['split_pairs']['train'];ix=slice(0,2);out=model(store.batch(train['drug'][ix],train['protein'][ix]));assert torch.isfinite(out).all()
    dump(OUT/'preflight.json',{'passed':True,'cache_sha256':sha(SOURCE_OUT/'cache/warm_811.pt'),'source_adapter_sha256':sha(SOURCE_OUT/'adapter/best.pt'),
         'train':24044,'val':3005,'test_sealed':3007,'finished':time.time()});print('PREFLIGHT_PASSED',flush=True)


def worker():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True);lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stages=['calibrate','structure','rnc','final'];env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(protocol()['gpu']),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for i,stage in enumerate(stages):
        free=int(subprocess.check_output(['nvidia-smi','-i',str(protocol()['gpu']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<30000: raise RuntimeError(f'GPU {protocol()["gpu"]} has only {free} MiB free')
        logpath=OUT/'logs'/f'{stage}.log'
        with logpath.open('a') as log:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),stage],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
            dump(OUT/'status.json',{'state':'running','stage':stage,'stage_index':i+1,'total_stages':len(stages),'worker_pid':os.getpid(),
                 'child_pid':proc.pid,'gpu':protocol()['gpu'],'log':str(logpath),'updated':time.time()});code=proc.wait()
        if code:
            dump(OUT/'status.json',{'state':'failed','stage':stage,'exit_code':code,'worker_pid':os.getpid(),'child_pid':proc.pid,'log':str(logpath),'updated':time.time()})
            raise RuntimeError(f'{stage} failed with exit {code}')
    dump(OUT/'status.json',{'state':'completed','worker_pid':os.getpid(),'gpu':protocol()['gpu'],'finished':time.time()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','calibrate','structure','rnc','final','worker']);a=p.parse_args()
    {'preflight':preflight,'calibrate':stage_calibrate,'structure':stage_structure,'rnc':stage_rnc,'final':stage_final,'worker':worker}[a.action]()
