"""Isolated real-data base backward, full cache and miniature end-to-end test."""
import json
import os
from pathlib import Path
import time
H=Path(__file__).resolve().parent
os.environ['WARM_RUN_ROOT']=str(H/'smoke_output')
os.environ['WARM_SEED']='42'
os.environ['WARM_GPU']='1'
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import torch
import run_fixed_pipeline as pipe

def main():
    torch.set_num_threads(4);pipe.seed_all(999);args=pipe.p13d_runner.arguments()
    dataset,train,val,_,_=pipe.base.build_dataloaders(args)
    assert len(dataset)==118254
    # Train-only random batch plus the largest training protein/drug examples.
    rng=np.random.default_rng(123);indices=rng.choice(train.indices,16,replace=False).tolist()
    for key,folder,field in [('protein_id','protein_3d_gvp','node_s'),('drug_id','drug_3d','x')]:
        ids=dataset.df.iloc[train.indices][key].unique()
        largest=max(ids,key=lambda x:len(torch.load(pipe.PROJECT/'data/processed/kiba'/folder/(x+'.pt'),weights_only=False)[field]))
        indices[0 if key=='protein_id' else 1]=next(i for i in train.indices if dataset.df.iloc[i][key]==largest)
    batch=pipe.base.move_batch_to_device(pipe.base.mdta_collate_fn_p13d([dataset[i] for i in indices]),torch.device('cuda'))
    model=pipe.base.build_model(args,torch.device('cuda'));optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    torch.cuda.reset_peak_memory_stats();started=time.time();pred=model(batch);loss=(pred-batch['label']).square().mean()
    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    report={'base_step_loss':float(loss.detach()),'base_step_seconds':time.time()-started,'base_peak_mib':torch.cuda.max_memory_allocated()/1024**2}
    pipe.save(pipe.CHECKPOINT,{'model_state_dict':model.state_dict()})
    del model,optimizer,batch,pred,loss;torch.cuda.empty_cache()
    pipe.preflight();pipe.build_cache()
    cache=pipe.tune.load_cache()
    assert np.isnan(cache['labels'][cache['pair_indices']<0]).all()
    assert np.isnan(cache['base'][cache['pair_indices']<0]).all()
    for part,n in [('train',64),('val',32),('test',32)]:
        cache['split_pairs'][part]={k:v[:n] for k,v in cache['split_pairs'][part].items()}
    cp=pipe.OUTPUT/'cache/warm_811.pt';pipe.save(cp,cache)
    audit_path=pipe.OUTPUT/'cache/audit.json';audit=json.loads(audit_path.read_text());audit['cache_sha256']=pipe.sha(cp);audit['smoke_only']=True;pipe.dump(audit_path,audit)
    test=cache['split_pairs']['test'];base_metrics=pipe.metric(cache['base'][test['drug'],test['protein']],cache['labels'][test['drug'],test['protein']])
    pipe.dump(pipe.P13D_OUTPUT/'result.json',{'test_metrics':base_metrics,'smoke_only':True})
    original=pipe.protocol
    def small():
        p=original();p['screening'].update(max_epochs=1,patience=1,eval_batch_size=16)
        return p
    pipe.protocol=small;pipe.tune.protocol=small
    pipe.train_and_evaluate()
    result=json.loads((pipe.OUTPUT/'result.json').read_text())
    assert result['complete'] and result['label_poisoning_invariance'] and result['raw_replay_matches_original']
    report.update(passed=True,full_feature_cache=True,sparse_missing_pairs_remain_nan=True,miniature_end_to_end=True,label_poisoning_invariance=True,
                  smoke_root=str(pipe.RUN_ROOT),not_scientific_results=True,finished=time.time())
    pipe.dump(H/'smoke_result.json',report);print('SMOKE_PASSED',json.dumps(report),flush=True)
if __name__=='__main__':main()
