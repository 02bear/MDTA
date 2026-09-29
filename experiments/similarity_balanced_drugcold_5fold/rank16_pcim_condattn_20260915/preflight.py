"""Behavioral preflight: equivalence, causality, gradients, resume, and real memory."""
import copy
import importlib.util
import json
import time

import numpy as np
import torch

from common import (OUT, CACHE_SOURCE, PARENT, FeatureStore, R1, dump, load_protocol,
                    restore_rng, rng_state, seed_all)
from pair_interaction import PairResidual


def synthetic(batch=4):
    device='cuda'
    generator=torch.Generator(device=device).manual_seed(71)
    return {'atoms':torch.randn(batch,5,128,generator=generator,device=device),
            'residues':torch.randn(batch,7,128,generator=generator,device=device),
            'aa':torch.randint(0,21,(batch,7),generator=generator,device=device),
            'atom_mask':torch.tensor([[1,1,1,0,0],[1,1,1,1,1]]*(batch//2),device=device,dtype=torch.bool),
            'residue_mask':torch.tensor([[1,1,1,1,0,0,0],[1,1,1,1,1,1,1]]*(batch//2),device=device,dtype=torch.bool),
            'drug_global':torch.randn(batch,128,generator=generator,device=device),
            'protein_global':torch.randn(batch,128,generator=generator,device=device)}


def parent_class():
    spec=importlib.util.spec_from_file_location('coverage_parent_pair_interaction',PARENT/'pair_interaction.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.PairResidual


def step(model,optimizer,b,target):
    model.train();optimizer.zero_grad(set_to_none=True)
    loss=(model(b)-target).square().mean();loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()),float(norm)


def model_checks():
    b=synthetic()
    shared=torch.cat([b['drug_global'].mean(0),b['protein_global'].mean(0)])
    Parent=parent_class()
    seed_all(42);parent=Parent('pair_graph',selection='coverage',coverage_budget=32).cuda()
    seed_all(42);baseline=PairResidual(attention_mode='none').cuda()
    seed_all(42);dynamic=PairResidual(attention_mode='dynamic').cuda()
    seed_all(42);fixed=PairResidual(attention_mode='shared',shared_global_pair=shared).cuda()

    parent_state=parent.state_dict();baseline_state=baseline.state_dict()
    assert parent_state.keys()==baseline_state.keys()
    assert all(torch.equal(parent_state[k],baseline_state[k]) for k in parent_state)
    dynamic_common={k:v for k,v in dynamic.state_dict().items() if 'condition_bias' not in k}
    assert baseline_state.keys()==dynamic_common.keys()
    assert all(torch.equal(baseline_state[k],dynamic_common[k]) for k in baseline_state)
    fixed_common={k:v for k,v in fixed.state_dict().items()
                  if 'condition_bias' not in k and k!='shared_global_pair'}
    assert baseline_state.keys()==fixed_common.keys()
    assert all(torch.equal(baseline_state[k],fixed_common[k]) for k in baseline_state)

    for model in [parent,baseline,dynamic,fixed]: model.eval()
    assert torch.equal(parent(b),baseline(b))
    assert torch.equal(baseline(b),dynamic(b))
    assert torch.equal(baseline(b),fixed(b))
    with torch.no_grad():
        parent.output.weight.normal_(0,.1)
        for model in [baseline,dynamic,fixed]: model.output.weight.copy_(parent.output.weight)
    # Readout draws are aligned too, so zero conditional projection preserves behavior.
    assert torch.equal(parent(b),baseline(b))
    assert torch.equal(baseline(b),dynamic(b))
    assert torch.equal(baseline(b),fixed(b))

    with torch.no_grad():
        dynamic.graph.condition_bias.weight.normal_(0,.08)
        fixed.graph.condition_bias.weight.copy_(dynamic.graph.condition_bias.weight)
    dynamic.eval();fixed.eval()
    clean,info=dynamic(b,diagnostics=True)
    off=dynamic(b,attention_override='off')
    shuffled=dynamic(b,attention_override='shuffle')
    assert float(info['condition_score_rms'].mean())>0
    assert float(info['attention_condition_change_rms'].mean())>0
    assert float((clean-off).abs().max())>1e-8
    assert float((clean-shuffled).abs().max())>1e-8
    fixed_clean=fixed(b)
    assert torch.equal(fixed_clean,fixed(b,attention_override='shuffle'))
    assert float((fixed_clean-fixed(b,attention_override='off')).abs().max())>1e-8

    # The new zero-initialized projection receives signal on the first real update.
    seed_all(81);grad_model=PairResidual(attention_mode='dynamic').cuda()
    with torch.no_grad(): grad_model.output.weight.normal_(0,.1)
    grad_model.train();target=torch.tensor([.1,-.2,.05,-.15],device='cuda')
    loss=(grad_model(b)-target).square().mean();loss.backward()
    grad=grad_model.graph.condition_bias.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0

    # Optimizer and RNG resume must reproduce the next stochastic step exactly.
    optimizer=torch.optim.AdamW(grad_model.parameters(),lr=3e-4)
    optimizer.zero_grad(set_to_none=True);optimizer.step()
    saved_model=copy.deepcopy(grad_model.state_dict());saved_opt=copy.deepcopy(optimizer.state_dict())
    state=rng_state();expected,_=step(grad_model,optimizer,b,target)
    resumed=PairResidual(attention_mode='dynamic').cuda();resumed.load_state_dict(saved_model)
    resumed_optimizer=torch.optim.AdamW(resumed.parameters(),lr=3e-4);resumed_optimizer.load_state_dict(saved_opt)
    restore_rng(state);actual,_=step(resumed,resumed_optimizer,b,target)
    assert expected==actual
    resume_error=max(float((x-y).abs().max()) for x,y in zip(grad_model.parameters(),resumed.parameters()))
    assert resume_error<1e-7,resume_error
    return {
        'parent_baseline_state_exact':True,
        'zero_init_baseline_dynamic_shared_output_exact':True,
        'dynamic_condition_changes_scores_attention_and_output':True,
        'dynamic_shuffle_changes_output':True,
        'shared_shuffle_is_exact_control':True,
        'condition_projection_first_step_gradient_nonzero':True,
        'resume_loss_exact':True,
        'resume_parameter_max_error':resume_error,
        'parameters':{name:sum(p.numel() for p in model.parameters()) for name,model in
                      [('baseline',baseline),('dynamic',dynamic),('shared',fixed)]},
        'conditional_parameter_increment':sum(p.numel() for p in dynamic.parameters())-
                                          sum(p.numel() for p in baseline.parameters())}


def main():
    torch.set_num_threads(4)
    checks=model_checks()
    assert checks['conditional_parameter_increment']==544
    cache=torch.load(CACHE_SOURCE/'fold_1.pt',map_location='cpu',weights_only=False)
    store=FeatureStore(cache)
    train=cache['split_drugs']['train']
    p=np.array(sorted(range(len(cache['proteins'])),key=lambda x:store.protein_length[x],reverse=True)[:16])
    d=np.array([train[i%len(train)] for i in range(16)])
    b=store.batch(d,p)
    model=PairResidual(attention_mode='dynamic').cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    seed_all(999);torch.cuda.reset_peak_memory_stats();begun=time.time()
    target=torch.as_tensor(cache['labels'][d,p]-cache['base'][d,p],device='cuda',dtype=torch.float32)
    for _ in range(5): step(model,optimizer,b,target)
    torch.cuda.synchronize()
    assert model.graph.condition_bias.weight.grad.abs().sum()>0
    r1=R1(cache)
    original=r1.correct(cache['base'],cache['split_drugs']['val'])[0]
    disabled=r1.correct(cache['base']+0*np.ones_like(cache['base']),cache['split_drugs']['val'])[0]
    assert np.array_equal(original,disabled)
    shifted=cache['base'].copy();shifted[train]+=.1
    assert not np.array_equal(r1.correct(shifted,cache['split_drugs']['val'])[0],original)
    cfg=load_protocol()
    result={'passed':True,'checks':checks,'formula':cfg['conditional_attention']['formula'],
            'real_max_length_batch':{'atoms_shape':list(b['atoms'].shape),
                'residues_shape':list(b['residues'].shape),'five_steps_seconds':time.time()-begun,
                'peak_cuda_mib':torch.cuda.max_memory_allocated()/1024**2},
            'disabled_r1_exact':True,'r1_rebuild_responds_to_combined_prediction':True,
            'timestamp':time.time()}
    dump(OUT/'preflight.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
