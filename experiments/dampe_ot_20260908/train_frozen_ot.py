"""F0-F4: frozen encoders, newly initialized original fusion/head, val-only pilot.

Uses the actual baseline training/evaluation functions and checkpoint selection
rule. Cached features avoid repeatedly executing fixed EGNNs. No test access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from common import OUTPUT, dump, sha256, assert_sources
from model_frozen_ot import FrozenOTHead
from train_p13d_earlystop import set_seed, train_one_epoch, evaluate


class CachedPairs(Dataset):
    def __init__(self, cache, part):
        pairs = cache['pairs'][part]
        self.features = {'label': pairs['label']}
        for kind in ['drug','protein']:
            entities = cache['entities'][kind]
            lookup = {eid:k for k,eid in enumerate(entities['ids'])}
            rows = torch.tensor([lookup[eid] for eid in pairs[kind+'_id']],dtype=torch.long)
            self.features[kind+'_1d'] = entities['h1'][rows]
            self.features[kind+'_3d'] = entities['h3'][rows]
        self.pairs = pairs

    def __len__(self):
        return len(self.features['label'])

    def __getitem__(self, i):
        return {k:v[i] for k,v in self.features.items()}


def mappings(root, mode, dim):
    identity = torch.eye(dim)
    if mode=='F0': return identity, identity
    if mode=='F4': return identity/dim, identity
    name='drug_shuffle' if mode=='F2' else 'drug'
    obj=torch.load(root/'ot'/(name+'_ot_matrix.pt'),map_location='cpu',weights_only=False)
    t=obj['T'].float()
    return (t if mode=='F3' else t*t.shape[1]), identity


def parameters_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def make_loader(dataset, seed, batch_size, shuffle):
    # Isolated generator pairs the exact shuffle stream across variants and
    # decouples it from random operations inside diagnostics or model creation.
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=0,
                      generator=torch.Generator().manual_seed(seed),pin_memory=False)


def train(a):
    torch.set_num_threads(1)
    root=OUTPUT/f'fold_{a.fold}'
    cachefile=root/'cache/features.pt'
    cache=torch.load(cachefile,map_location='cpu',weights_only=False)
    assert_sources(cache)
    base=cache['baseline_args']
    ot_manifest=json.loads((root/'ot/manifest.json').read_text())
    if ot_manifest['cache_sha256']!=sha256(cachefile):
        raise RuntimeError('OT/cache mismatch')
    for name,digest in ot_manifest['matrices'].items():
        if sha256(root/'ot'/(name+'_ot_matrix.pt'))!=digest:
            raise RuntimeError('OT matrix changed after fitting')
    suffix='smoke' if a.smoke else 'runs'
    out=root/suffix/f'{a.mode}_seed{a.seed}'
    out.mkdir(parents=True,exist_ok=False)
    set_seed(a.seed)
    dmap,pmap=mappings(root,a.mode,base['hidden_dim'])
    model=FrozenOTHead(dmap,pmap,base['hidden_dim'],base['dropout']).to(a.device)
    init_hash=parameters_hash(model)
    train_ds,val_ds=CachedPairs(cache,'train'),CachedPairs(cache,'val')
    train_loader=make_loader(train_ds,a.seed,base['batch_size'],True)
    val_loader=make_loader(val_ds,a.seed,base['batch_size'],False)
    # Audit the first epoch sample ordering with a separate generator.
    order_probe=make_loader(train_ds,a.seed,base['batch_size'],True)
    # DataLoader iterator consumes one base_seed draw before the first sample.
    torch.empty((),dtype=torch.int64).random_(generator=order_probe.generator)
    order=np.array([i for batch in order_probe.batch_sampler for i in batch],dtype=np.int64)
    optimizer=torch.optim.Adam(model.parameters(),lr=base['lr'],weight_decay=base['weight_decay'])
    criterion=torch.nn.MSELoss()
    config=dict(mode=a.mode,seed=a.seed,fold=a.fold,protocol='frozen_encoder_new_original_fusion_head',
        device=a.device,epochs=a.epochs if a.smoke else base['epochs'],batch_size=base['batch_size'],
        lr=base['lr'],weight_decay=base['weight_decay'],optimizer='Adam',loss='MSE',scheduler=None,
        early_stop_patience=base['early_stop_patience'],early_stop_min_delta=base['early_stop_min_delta'],
        hidden_dim=base['hidden_dim'],dropout=base['dropout'],initial_trainable_parameters_sha256=init_hash,
        first_epoch_sampler_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        cache_sha256=sha256(cachefile),source_sha256=cache['source_sha256'],
        checkpoint_sha256=cache['checkpoint_sha256'],split_sha256=cache['split_sha256'],
        test_evaluated=False,ot_sha256=ot_manifest['matrices'],
        epsilon=ot_manifest['epsilon'],encoder_dropout='disabled_eval_cache',
        train_loop='train_p13d_earlystop.train_one_epoch',validation_loop='train_p13d_earlystop.evaluate',
        seed_scope='downstream_initialization_and_shuffle_only_fixed_seed42_encoder',smoke=a.smoke)
    dump(out/'config.json',config)
    print('CONFIG',json.dumps(config),flush=True)
    best=float('inf');best_epoch=-1;counter=0;history=[];started=time.monotonic()
    for epoch in range(1,config['epochs']+1):
        tm=train_one_epoch(model,train_loader,criterion,optimizer,torch.device(a.device),log_interval=100000)
        vm=evaluate(model,val_loader,criterion,torch.device(a.device))
        if not all(np.isfinite(v) for v in list(tm.values())+list(vm.values())):
            raise RuntimeError('Nonfinite metric; refusing checkpoint')
        improved=vm['rmse']<best-base['early_stop_min_delta']
        history.append(dict(epoch=epoch,train=tm,val=vm,elapsed_seconds=time.monotonic()-started))
        state=dict(epoch=epoch,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                   train_metrics=tm,val_metrics=vm,config=config)
        torch.save(state,out/'latest_model.pt')
        if improved:
            best=vm['rmse'];best_epoch=epoch;counter=0
            torch.save(state,out/'best_model.pt')
        else:
            counter+=1
        dump(out/'history.json',history)
        dump(out/'progress.json',dict(epoch=epoch,best_epoch=best_epoch,best_val_rmse=best,
            no_improvement=counter,status='training',elapsed_seconds=time.monotonic()-started))
        print(f'{a.mode} seed={a.seed} epoch={epoch} train_mse={tm["mse"]:.6f} val_mse={vm["mse"]:.6f} '
              f'best_epoch={best_epoch} counter={counter} seconds={time.monotonic()-started:.1f}',flush=True)
        if base['early_stop_patience']>0 and counter>=base['early_stop_patience']:
            break
    # Reload the exact winning model including mapping buffers and verify metrics.
    ck=torch.load(out/'best_model.pt',map_location=a.device,weights_only=False)
    model.load_state_dict(ck['model_state_dict'],strict=True)
    verified=evaluate(model,val_loader,criterion,torch.device(a.device))
    for key,value in verified.items():
        if abs(value-ck['val_metrics'][key])>1e-6:
            raise RuntimeError('Checkpoint metric replay failed '+key)
    preds=[]
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            preds.append(model({k:v.to(a.device) for k,v in batch.items()}).cpu())
    np.savez_compressed(out/'best_val_predictions.npz',pred=torch.cat(preds).numpy().reshape(-1),
        label=val_ds.features['label'].numpy().reshape(-1),
        drug_id=np.asarray(val_ds.pairs['drug_id'],dtype=str),
        protein_id=np.asarray(val_ds.pairs['protein_id'],dtype=str),
        pair_index=np.asarray(val_ds.pairs['indices']))
    summary=dict(mode=a.mode,seed=a.seed,fold=a.fold,best_epoch=best_epoch,
        best_val_metrics=verified,epochs_completed=epoch,elapsed_seconds=time.monotonic()-started,
        initial_trainable_parameters_sha256=init_hash,
        checkpoint_replay_verified=True,test_evaluated=False,complete=True)
    dump(out/'summary.json',summary)
    dump(out/'progress.json',dict(status='complete',**summary))
    print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=1)
    p.add_argument('--mode',choices=['F0','F1','F2','F3','F4'],required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cpu')
    p.add_argument('--smoke',action='store_true');p.add_argument('--epochs',type=int,default=2)
    train(p.parse_args())
