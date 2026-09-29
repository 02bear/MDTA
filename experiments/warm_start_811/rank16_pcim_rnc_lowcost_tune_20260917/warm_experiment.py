"""Leakage-safe DAVIS warm-start Rank16 + PCIM + standard RNC 0.01 + R1."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PROJECT = Path('/data1/ztx/MyModel-MDTA')
OUT = PROJECT/'outputs/Refine_experiment/davis/warm_start/random_pair_811_seed42/rank16_pcim_pair_rnc_20260917'
SPLIT = PROJECT/'data/splits/davis_fixed_split_811_full.json'
sys.path.insert(0, str(HERE/'source'))
sys.path.insert(0, str(HERE))

from pair_interaction import PairResidual
from rnc_loss import StratifiedPairSampler, rank_n_contrast_loss


def sha(path):
    dig = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            dig.update(block)
    return dig.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        torch.save(obj, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch']); torch.cuda.set_rng_state_all(s['cuda'])


def modules():
    spec = importlib.util.spec_from_file_location('warm_rank16', HERE/'model_impl.py')
    h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    return h, h.base


def base_args():
    cfg = protocol()['backbone']
    return SimpleNamespace(
        pairs_csv='data/raw/davis/pairs.csv',
        drug_1d_dir='data/processed/davis/drug_1d_chemberta2',
        drug_2d_dir='data/processed/davis/drug_2d',
        drug_3d_dir='data/processed/davis/drug_3d',
        protein_1d_dir='data/processed/davis/protein_1d_esm2',
        protein_3d_dir='data/processed/davis/protein_3d_gvp',
        split_json=str(SPLIT), output_dir=str(OUT/'base_rank16'), seed=protocol()['seed'],
        train_ratio=0.8, batch_size=cfg['batch_size'], num_workers=0,
        epochs=cfg['epochs'], lr=cfg['lr'], weight_decay=cfg['weight_decay'],
        early_stop_patience=cfg['early_stop_patience'],
        early_stop_min_delta=cfg['early_stop_min_delta'],
        drug_1d_in_dim=768, drug_3d_node_in_dim=10,
        hidden_dim=cfg['hidden_dim'], dropout=cfg['dropout'])


@torch.no_grad()
def evaluate_base(model, loader, criterion, device, base_module):
    model.eval(); preds=[]; targets=[]; total=0.0
    for batch in loader:
        batch = base_module.move_batch_to_device(batch, device)
        pred = model(batch).reshape(-1); target = batch['label'].reshape(-1)
        total += float(criterion(pred, target)) * len(target)
        preds.append(pred.cpu()); targets.append(target.cpu())
    pred=torch.cat(preds); target=torch.cat(targets)
    result=base_module.compute_regression_metrics(pred, target)
    result['loss']=total/len(loader.dataset)
    return result, pred.numpy().astype(np.float64), target.numpy().astype(np.float64)


def train_base():
    cfg=protocol(); args=base_args(); h,t=modules(); seed_all(args.seed)
    folder=OUT/'base_rank16'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training_complete.json').exists():
        print('BASE_REUSED', flush=True); return
    device=torch.device('cuda')
    dataset,tr,va,train_loader,val_loader=t.build_dataloaders(args)
    split=json.loads(SPLIT.read_text())
    assert list(tr.indices)==split['train_indices'] and list(va.indices)==split['val_indices']
    assert len(tr)==24044 and len(va)==3005 and len(split['test_indices'])==3007
    model=h.build_model(args,device)
    criterion=torch.nn.MSELoss()
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    history=[]; best_rmse=float('inf'); best_epoch=-1; stale=0; best_train=None; best_val=None; start=1
    latest=folder/'latest_model.pt'
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model_state_dict'],strict=True); opt.load_state_dict(ck['optimizer_state_dict'])
        history=ck['history']; best_rmse=ck['best_val_rmse']; best_epoch=ck['best_epoch']; stale=ck['stale']
        best_train=ck['best_train']; best_val=ck['best_val']; start=ck['epoch']+1; restore_rng(ck['rng'])
        print('BASE_RESUME',start,flush=True)
    print('BASE_START',json.dumps({'train':len(tr),'val':len(va),'test_sealed':len(split['test_indices']),
          'parameters':sum(p.numel() for p in model.parameters())}),flush=True)
    for epoch in range(start,args.epochs+1):
        if stale>=args.early_stop_patience: break
        began=time.time()
        tm=t.train_one_epoch(model,train_loader,criterion,opt,device,log_interval=200)
        vm,vpred,vtrue=evaluate_base(model,val_loader,criterion,device,t)
        assert np.isfinite([tm['loss'],vm['mse'],vm['rmse']]).all()
        improved=vm['rmse'] < best_rmse-args.early_stop_min_delta
        if improved:
            best_rmse=vm['rmse'];best_epoch=epoch;stale=0;best_train=dict(tm);best_val=dict(vm)
        else: stale+=1
        rec={'epoch':epoch,'train':tm,'val':vm,'best_epoch':best_epoch,'stale':stale,'seconds':time.time()-began}
        history.append(rec)
        ck={'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),
            'args':vars(args),'history':history,'best_val_rmse':best_rmse,'best_epoch':best_epoch,
            'stale':stale,'best_train':best_train,'best_val':best_val,'rng':rng_state(),'protocol':cfg}
        if improved:
            save(folder/'best_model.pt',ck)
            np.savez_compressed(folder/'best_val_predictions.npz',indices=np.asarray(va.indices),y_pred=vpred,y_true=vtrue)
        save(latest,ck); dump(folder/'history.json',history)
        dump(folder/'progress.json',{'state':'training',**rec,'updated':time.time()})
        print('BASE_EPOCH',json.dumps(rec),flush=True)
    assert best_epoch>0
    dump(folder/'best_summary.json',{'best_epoch':best_epoch,'best_train_metrics':best_train,'best_val_metrics':best_val})
    dump(folder/'training_complete.json',{'best_epoch':best_epoch,'last_epoch':history[-1]['epoch'],
         'checkpoint_sha256':sha(folder/'best_model.pt'),'finished':time.time()})
    print('BASE_FINISHED',best_epoch,flush=True)


class FeatureStore:
    def __init__(self, cache, device='cuda'):
        from torch.nn.utils.rnn import pad_sequence
        self.device=device
        self.atom_length=[len(x) for x in cache['atoms']]; self.protein_length=[len(x) for x in cache['residues']]
        self.atoms=pad_sequence(cache['atoms'],batch_first=True).to(device)
        self.residues=pad_sequence(cache['residues'],batch_first=True).to(device)
        self.aa=pad_sequence(cache['aa'],batch_first=True,padding_value=20).to(device)
        self.drug_global=cache['drug_global'].to(device); self.protein_global=cache['protein_global'].to(device)
        self.atom_mask=torch.arange(self.atoms.size(1),device=device)[None] < torch.tensor(self.atom_length,device=device)[:,None]
        self.residue_mask=torch.arange(self.residues.size(1),device=device)[None] < torch.tensor(self.protein_length,device=device)[:,None]

    def batch(self, drug, protein):
        d=torch.as_tensor(drug,device=self.device); p=torch.as_tensor(protein,device=self.device)
        nd=max(self.atom_length[int(x)] for x in drug); nr=max(self.protein_length[int(x)] for x in protein)
        return {'drug_global':self.drug_global[d],'protein_global':self.protein_global[p],
                'atoms':self.atoms[d,:nd],'atom_mask':self.atom_mask[d,:nd],
                'residues':self.residues[p,:nr],'residue_mask':self.residue_mask[p,:nr],'aa':self.aa[p,:nr]}


def build_cache():
    destination=OUT/'cache/warm_811.pt'
    if destination.exists() and (OUT/'cache/audit.json').exists():
        audit=json.loads((OUT/'cache/audit.json').read_text()); assert sha(destination)==audit['cache_sha256']
        print('CACHE_REUSED',flush=True); return
    h,t=modules(); ckpath=OUT/'base_rank16/best_model.pt'
    assert ckpath.exists(), 'Rank16 backbone is not complete'
    ck=torch.load(ckpath,map_location='cpu',weights_only=False); args=SimpleNamespace(**ck['args'])
    seed_all(protocol()['seed']); model=h.build_model(args,torch.device('cuda'))
    model.load_state_dict(ck['model_state_dict'],strict=True); model.eval().requires_grad_(False)
    dataset,tr,va,_,_=t.build_dataloaders(args); df=dataset.df.copy()
    raw=pd.read_csv(PROJECT/args.pairs_csv,dtype={'drug_id':str,'protein_id':str})
    df['drug_id']=df.drug_id.astype(str);df['protein_id']=df.protein_id.astype(str)
    assert len(df)==30056 and df[['drug_id','protein_id']].equals(raw[['drug_id','protein_id']])
    sim=np.load(HERE/'entity_similarities.npz',allow_pickle=True)
    drugs=list(map(str,sim['drug_ids'].tolist()));proteins=list(map(str,sim['protein_ids'].tolist()))
    di={x:i for i,x in enumerate(drugs)};pi={x:i for i,x in enumerate(proteins)}
    rows=np.full((len(drugs),len(proteins)),-1,dtype=np.int64)
    row_d=np.empty(len(df),dtype=np.int64);row_p=np.empty(len(df),dtype=np.int64)
    for i,r in enumerate(df.itertuples(index=False)):
        d,p=di[r.drug_id],pi[r.protein_id];rows[d,p]=i;row_d[i]=d;row_p[i]=p
    assert (rows>=0).all() and np.unique(rows).size==len(df)
    labels=np.empty_like(rows,dtype=np.float64);labels[row_d,row_p]=df.label.to_numpy(dtype=np.float64)
    AA3=['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR']
    aa_map={name:i for i,name in enumerate(AA3)};aa_map['MSE']=aa_map['MET']
    atoms=[];residues=[];aa=[];dglobal=[];pglobal=[]
    with torch.inference_mode():
        for index,d in enumerate(drugs):
            obj=torch.load(PROJECT/args.drug_3d_dir/f'{d}.pt',map_location='cpu',weights_only=False)
            x1=torch.load(PROJECT/args.drug_1d_dir/f'{d}.pt',map_location='cpu',weights_only=False)['mean'].float()[None].cuda()
            graph={k:obj[k].cuda() for k in ['x','pos','edge_index']};graph['x']=graph['x'].float();graph['pos']=graph['pos'].float()
            graph['batch']=torch.zeros(len(obj['x']),dtype=torch.long,device='cuda')
            out=model.drug_3d_encoder(graph,return_node=True);atoms.append(out['node_feat'].cpu().clone())
            dglobal.append(model.drug_fusion([model.drug_1d_encoder(x1),out['graph_feat']]).cpu().clone()[0])
            if index%20==0: print('CACHE_DRUG',index,len(drugs),flush=True)
        for index,p in enumerate(proteins):
            obj=torch.load(PROJECT/args.protein_3d_dir/f'{p}.pt',map_location='cpu',weights_only=False)
            types=torch.tensor([aa_map.get(str(m['resname']).upper(),20) for m in obj['residue_meta']],dtype=torch.long);aa.append(types)
            x1=torch.load(PROJECT/args.protein_1d_dir/f'{p}.pt',map_location='cpu',weights_only=False)['mean'].float()[None].cuda()
            graph={k:obj[k].cuda() for k in ['node_s','node_v','coords','edge_index','edge_s','edge_v']}
            graph['batch']=torch.zeros(len(types),dtype=torch.long,device='cuda')
            out=model.protein_3d_encoder(graph,return_node=True);residues.append(out['node_feat'].cpu().clone())
            pglobal.append(model.protein_fusion([model.protein_1d_encoder(x1),out['graph_feat']]).cpu().clone()[0])
            if index%50==0: print('CACHE_PROTEIN',index,len(proteins),flush=True)
        dg,pg=torch.stack(dglobal),torch.stack(pglobal);base=np.empty_like(labels)
        for start in range(0,len(df),64):
            dd=row_d[start:start+64];pp=row_p[start:start+64]
            pred=model.decoder(torch.cat([dg[dd].cuda(),pg[pp].cuda()],-1)).reshape(-1).cpu().numpy()
            base[dd,pp]=pred
    replay=np.load(OUT/'base_rank16/best_val_predictions.npz',allow_pickle=True)
    idx=replay['indices'];error=float(np.max(np.abs(base[row_d[idx],row_p[idx]]-replay['y_pred'])))
    assert error<1e-4,error
    split=json.loads(SPLIT.read_text());split_pairs={}
    for part in ['train','val','test']:
        ix=np.asarray(split[part+'_indices'],dtype=np.int64)
        split_pairs[part]={'index':ix,'drug':row_d[ix],'protein':row_p[ix]}
    assert not set(split_pairs['train']['index'])&set(split_pairs['val']['index'])
    assert not set(split_pairs['train']['index'])&set(split_pairs['test']['index'])
    data={'drugs':drugs,'proteins':proteins,'pair_indices':rows,'row_drug':row_d,'row_protein':row_p,
          'labels':labels,'base':base,'atoms':atoms,'residues':residues,'aa':aa,
          'drug_global':dg,'protein_global':pg,'split_pairs':split_pairs,
          'similarity':sim['drug_similarity'].astype(np.float64),'checkpoint':str(ckpath),
          'checkpoint_sha256':sha(ckpath),'split_sha256':sha(SPLIT)}
    save(destination,data)
    dump(OUT/'cache/audit.json',{'passed':True,'cache_sha256':sha(destination),'split_sha256':sha(SPLIT),
         'checkpoint_sha256':sha(ckpath),'validation_replay_max_error':error,
         'train':len(split_pairs['train']['index']),'val':len(split_pairs['val']['index']),
         'test':len(split_pairs['test']['index'])})
    print('CACHE_FINISHED',error,flush=True)


def masked_stats(qd,qp,residual,mask,similarity,cfg):
    mean=np.zeros(len(qd),dtype=np.float64);var=np.zeros(len(qd),dtype=np.float64);support=np.zeros(len(qd),dtype=np.float64)
    for i,(d,p) in enumerate(zip(qd,qp)):
        refs=np.flatnonzero(mask[:,p]); sims=similarity[d,refs]
        order=np.argsort(sims)[::-1][:min(cfg['k_drug'],len(refs))]
        refs=refs[order];sims=sims[order];keep=sims>=cfg['min_drug_similarity'];refs=refs[keep];sims=sims[keep]
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
    residual=np.zeros_like(prediction,dtype=np.float64)
    residual[train['drug'],train['protein']]=labels[train['drug'],train['protein']]-prediction[train['drug'],train['protein']]
    mean,var,support=masked_stats(query['drug'],query['protein'],residual,mask,similarity,cfg)
    base=prediction[query['drug'],query['protein']]
    return apply_r1(base,mean,var,support,cfg)


def select_r1(cache):
    path=OUT/'r1_config.json'
    if path.exists(): return json.loads(path.read_text())
    cfg=protocol()['r1'];train=cache['split_pairs']['train'];val=cache['split_pairs']['val']
    mask=np.zeros_like(cache['base'],dtype=bool);mask[train['drug'],train['protein']]=True
    residual=np.zeros_like(cache['base']);residual[train['drug'],train['protein']]=cache['labels'][train['drug'],train['protein']]-cache['base'][train['drug'],train['protein']]
    target=cache['labels'][val['drug'],val['protein']];base=cache['base'][val['drug'],val['protein']]
    # A complete no-op candidate keeps evaluation safe when validation selects
    # the R1 boundary (i.e. residual correction has no measurable benefit).
    best={
        'mse':float(np.mean((base-target)**2)), 'boundary':True,
        'gamma':1.0, 'k_drug':4, 'min_neighbors':0, 'tau':1.0,
        'beta':0.0, 'clip':0.25, 'scale':0.0,
    }
    for gamma in cfg['gamma']:
      for k in cfg['k_drug']:
       for minimum in cfg['min_drug_similarity']:
        drug_cfg={'gamma':gamma,'k_drug':k,'min_drug_similarity':minimum}
        mean,var,support=masked_stats(val['drug'],val['protein'],residual,mask,cache['similarity'],drug_cfg)
        for tau in cfg['tau']:
         for beta in cfg['beta']:
          for clip in cfg['clip']:
           for scale in cfg['scale']:
            item={**drug_cfg,'tau':tau,'beta':beta,'clip':clip,'scale':scale}
            pred,_,_=apply_r1(base,mean,var,support,item);mse=float(np.mean((pred-target)**2))
            if mse<best['mse']: best={**item,'mse':mse,'boundary':False}
    dump(path,{**best,'selection_data':'validation labels; residual reference labels restricted to training pairs',
         'baseline_validation_mse':float(np.mean((base-target)**2))})
    print('R1_SELECTED',json.dumps(best),flush=True);return json.loads(path.read_text())


@torch.no_grad()
def predict_pairs(model,store,pairs,batch_size):
    model.eval();out=[]
    for start in range(0,len(pairs['drug']),batch_size):
        out.append(model(store.batch(pairs['drug'][start:start+batch_size],pairs['protein'][start:start+batch_size])).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def metric(pred,label):
    _,t=modules()
    return t.compute_regression_metrics(torch.tensor(pred,dtype=torch.float64),torch.tensor(label,dtype=torch.float64))


def train_adapter():
    cfg=protocol();acfg=cfg['adapter'];rcfg=cfg['rnc'];dest=OUT/'adapter';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists(): print('ADAPTER_REUSED',flush=True);return
    cache=torch.load(OUT/'cache/warm_811.pt',map_location='cpu',weights_only=False)
    audit=json.loads((OUT/'cache/audit.json').read_text());assert sha(OUT/'cache/warm_811.pt')==audit['cache_sha256']
    train=cache['split_pairs']['train'];val=cache['split_pairs']['val'];test=cache['split_pairs']['test']
    r1cfg=select_r1(cache);seed_all(cfg['seed']);store=FeatureStore(cache)
    model=PairResidual('bidirectional',top_k=acfg['top_k'],max_delta=acfg['max_delta'],selection='coverage',
          coverage_budget=acfg['coverage_budget'],cross_max_scale=acfg['cross_max_scale'],enable_contrast=True,
          contrast_dim=rcfg['projection_dim']).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=acfg['lr'],weight_decay=acfg['weight_decay'])
    development=cache['labels'].copy();development[test['drug'],test['protein']]=np.nan
    baseline_val=r1_correct(cache['base'],development,train,val,cache['similarity'],r1cfg)[0]
    baseline_mse=float(np.mean((baseline_val-development[val['drug'],val['protein']])**2))
    best={'source':'rank16_r1','lambda':0.0,'mse':baseline_mse,'epoch':0};history=[];stale=0;start_epoch=1
    save(dest/'best.pt',{'model':model.state_dict(),'selection':best.copy(),'protocol':cfg})
    latest=dest/'latest.pt'
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        best=ck['best'];history=ck['history'];stale=ck['stale'];start_epoch=ck['epoch']+1;restore_rng(ck['rng'])
        print('ADAPTER_RESUME',start_epoch,flush=True)
    labels_train=development[train['drug'],train['protein']]
    sampler=StratifiedPairSampler(labels_train,batch_size=rcfg['batch_size'],high_count=rcfg['high_count'],
         mid_count=rcfg['mid_count'],high_threshold=rcfg['high_threshold'],mid_threshold=rcfg['mid_threshold'])
    print('ADAPTER_START',json.dumps({'train':len(labels_train),'val':len(val['index']),'test_sealed':len(test['index']),
          'parameters':sum(p.numel() for p in model.parameters()),'baseline_validation_mse':baseline_mse,
          'r1_config':r1cfg,'rnc_sampler':sampler.audit()}),flush=True)
    for epoch in range(start_epoch,acfg['epochs']+1):
        if stale>=acfg['patience']: break
        began=time.time();model.train();order=np.random.permutation(len(labels_train));losses=[];mses=[];rncs=[];norms=[]
        total_steps=(len(order)+acfg['batch_size']-1)//acfg['batch_size']
        for step,start in enumerate(range(0,len(order),acfg['batch_size'])):
            ix=order[start:start+acfg['batch_size']];d=train['drug'][ix];p=train['protein'][ix]
            target=torch.as_tensor(development[d,p]-cache['base'][d,p],dtype=torch.float32,device='cuda')
            opt.zero_grad(set_to_none=True);delta=model(store.batch(d,p));mse=(delta-target).square().mean()
            task=mse+acfg['residual_l2']*delta.square().mean();task.backward();rnc_value=0.0
            if (step+1)%rcfg['interval']==0:
                rix=sampler.sample();rd=train['drug'][rix];rp=train['protein'][rix];emb=[]
                for rstart in range(0,len(rix),rcfg['microbatch_size']):
                    _,z=model(store.batch(rd[rstart:rstart+rcfg['microbatch_size']],rp[rstart:rstart+rcfg['microbatch_size']]),return_contrast=True);emb.append(z)
                z=torch.cat(emb);aff=torch.as_tensor(labels_train[rix],dtype=torch.float32,device='cuda')
                rloss,_=rank_n_contrast_loss(z,aff,temperature=rcfg['temperature'],mode='standard',high_threshold=rcfg['high_threshold'])
                warm=min(1.0,((epoch-1)+(step+1)/total_steps)/rcfg['warmup_epochs'])
                weighted=rcfg['weight']*rcfg['interval']*warm*rloss;weighted.backward();rnc_value=float(rloss.detach());rncs.append(rnc_value)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),acfg['grad_clip'],error_if_nonfinite=True);opt.step()
            losses.append(float(task.detach())+rnc_value*rcfg['weight']);mses.append(float(mse.detach()));norms.append(float(norm))
            if step%200==0: print(f'ADAPTER_TRAIN epoch={epoch} step={step}/{total_steps} mse={mses[-1]:.6f}',flush=True)
        train_delta=predict_pairs(model,store,train,acfg['eval_batch_size']);val_delta=predict_pairs(model,store,val,acfg['eval_batch_size'])
        choices=[]
        for scale in acfg['lambdas']:
            pred=cache['base'].copy();pred[train['drug'],train['protein']]+=scale*train_delta;pred[val['drug'],val['protein']]+=scale*val_delta
            corrected=r1_correct(pred,development,train,val,cache['similarity'],r1cfg)[0]
            choices.append({'lambda':float(scale),'mse':float(np.mean((corrected-development[val['drug'],val['protein']])**2))})
        chosen=min(choices,key=lambda x:(x['mse'],x['lambda']));improved=chosen['mse']<best['mse']-acfg['min_delta']
        if improved:
            best={**chosen,'source':'pcim_rnc_r1','epoch':epoch};stale=0
            save(dest/'best.pt',{'model':model.state_dict(),'selection':best.copy(),'protocol':cfg})
        else: stale+=1
        rec={'epoch':epoch,'train_loss':float(np.mean(losses)),'train_mse':float(np.mean(mses)),
             'rnc_loss':float(np.mean(rncs)),'max_gradient_norm':max(norms),'validation_candidates':choices,
             'best':best.copy(),'stale':stale,'seconds':time.time()-began}
        history.append(rec);save(latest,{'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,
             'best':best,'history':history,'stale':stale,'rng':rng_state(),'protocol':cfg})
        dump(dest/'history.json',history);dump(dest/'progress.json',{'state':'training',**rec,'updated':time.time()})
        print('ADAPTER_EPOCH',json.dumps(rec),flush=True)
    dump(dest/'selection_locked.json',{'best':best,'r1_config':r1cfg,'selection_data':'validation only; test labels sealed',
         'finished':time.time()})
    selected=torch.load(dest/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected['model']);scale=best['lambda']
    train_delta=predict_pairs(model,store,train,acfg['eval_batch_size']);val_delta=predict_pairs(model,store,val,acfg['eval_batch_size']);test_delta=predict_pairs(model,store,test,acfg['eval_batch_size'])
    pred=cache['base'].copy();pred[train['drug'],train['protein']]+=scale*train_delta;pred[val['drug'],val['protein']]+=scale*val_delta;pred[test['drug'],test['protein']]+=scale*test_delta
    base_r1=r1_correct(cache['base'],cache['labels'],train,test,cache['similarity'],r1cfg)[0]
    corrected,correction,alpha=r1_correct(pred,cache['labels'],train,test,cache['similarity'],r1cfg)
    poisoned=cache['labels'].copy();poisoned[val['drug'],val['protein']]=np.nan;poisoned[test['drug'],test['protein']]=np.nan
    corrected2=r1_correct(pred,poisoned,train,test,cache['similarity'],r1cfg)[0];assert np.array_equal(corrected,corrected2)
    y=cache['labels'][test['drug'],test['protein']];raw_base=cache['base'][test['drug'],test['protein']];raw_new=pred[test['drug'],test['protein']]
    tests={'rank16':metric(raw_base,y),'rank16_R1':metric(base_r1,y),'rank16_pcim_rnc':metric(raw_new,y),'rank16_pcim_rnc_R1':metric(corrected,y)}
    high=y>=rcfg['high_threshold'];high_metrics={k:metric(v[high],y[high]) for k,v in {'rank16_R1':base_r1,'rank16_pcim_rnc_R1':corrected}.items()}
    table=pd.DataFrame({'pair_index':test['index'],'drug_id':[cache['drugs'][x] for x in test['drug']],
          'protein_id':[cache['proteins'][x] for x in test['protein']],'label':y,'rank16':raw_base,
          'rank16_R1':base_r1,'delta_raw':test_delta,'lambda':scale,'rank16_pcim_rnc':raw_new,
          'rank16_pcim_rnc_R1':corrected,'R1_correction':correction,'R1_alpha':alpha})
    table.to_csv(dest/'test_predictions.csv',index=False)
    result={'selection':best,'test':tests,'high_affinity_test':high_metrics,'high_affinity_count':int(high.sum()),
            'label_poisoning_invariance':True,'r1_config':r1cfg,'backbone_checkpoint':cache['checkpoint'],
            'backbone_sha256':cache['checkpoint_sha256'],'adapter_sha256':sha(dest/'best.pt'),'finished':time.time()}
    dump(dest/'result.json',result);dump(dest/'progress.json',{'state':'completed','selection':best,'updated':time.time()})
    (OUT/'REPORT.md').write_text('# DAVIS warm-start Rank16 + PCIM + standard RNC 0.01 + R1\n\n'+
        f'Selection: {best}\n\nTest metrics:\n\n```json\n{json.dumps(tests,indent=2)}\n```\n')
    print('TEST_RESULT',json.dumps(result),flush=True)


def preflight():
    cfg=protocol();split=json.loads(SPLIT.read_text());all_sets=[set(split[x+'_indices']) for x in ['train','val','test']]
    assert [len(x) for x in all_sets]==[24044,3005,3007]
    assert not all_sets[0]&all_sets[1] and not all_sets[0]&all_sets[2] and not all_sets[1]&all_sets[2]
    assert len(set.union(*all_sets))==30056
    h,t=modules();args=base_args();seed_all(args.seed);dataset,tr,va,train_loader,_=t.build_dataloaders(args)
    assert len(dataset)==30056 and len(tr)==24044 and len(va)==3005
    model=h.build_model(args,torch.device('cuda'));batch=next(iter(train_loader));batch=t.move_batch_to_device(batch,torch.device('cuda'))
    pred=model(batch).reshape(-1);loss=(pred-batch['label'].reshape(-1)).square().mean();loss.backward();assert torch.isfinite(loss)
    pair=PairResidual('bidirectional',top_k=cfg['adapter']['top_k'],selection='coverage',
        coverage_budget=cfg['adapter']['coverage_budget'],enable_contrast=True,contrast_dim=cfg['rnc']['projection_dim'])
    dump(OUT/'preflight.json',{'passed':True,'split_sha256':sha(SPLIT),'train':len(tr),'val':len(va),
         'test':len(split['test_indices']),'base_parameters':sum(p.numel() for p in model.parameters()),
         'adapter_parameters':sum(p.numel() for p in pair.parameters()),'smoke_loss':float(loss.detach()),'finished':time.time()})
    print('PREFLIGHT_PASSED',flush=True)


def worker():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=OUT/'status.json';tasks=['base','cache','adapter'];env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(protocol()['gpu']),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for position,action in enumerate(tasks):
        free=int(subprocess.check_output(['nvidia-smi','-i',str(protocol()['gpu']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<30000: raise RuntimeError(f'GPU {protocol()["gpu"]} has only {free} MiB free')
        logpath=OUT/f'logs/{action}.log'
        with logpath.open('a') as log:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),action],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
            dump(status,{'state':'running','stage':action,'task_index':position+1,'total_tasks':len(tasks),
                 'worker_pid':os.getpid(),'child_pid':proc.pid,'gpu':protocol()['gpu'],'log':str(logpath),'updated':time.time()})
            code=proc.wait()
        if code:
            dump(status,{'state':'failed','stage':action,'exit_code':code,'worker_pid':os.getpid(),'child_pid':proc.pid,
                 'gpu':protocol()['gpu'],'log':str(logpath),'updated':time.time()})
            raise RuntimeError(f'{action} failed with exit {code}; see {logpath}')
    dump(status,{'state':'completed','worker_pid':os.getpid(),'gpu':protocol()['gpu'],'finished':time.time()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','base','cache','adapter','worker']);args=parser.parse_args()
    {'preflight':preflight,'base':train_base,'cache':build_cache,'adapter':train_adapter,'worker':worker}[args.action]()
