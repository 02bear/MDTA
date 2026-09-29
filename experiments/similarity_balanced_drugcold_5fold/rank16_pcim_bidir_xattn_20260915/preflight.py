"""Preflight bidirectional cross-attention before any experiment run."""
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
    generator=torch.Generator(device='cuda').manual_seed(71)
    return {'atoms':torch.randn(batch,5,128,generator=generator,device='cuda'),
            'residues':torch.randn(batch,7,128,generator=generator,device='cuda'),
            'aa':torch.randint(0,21,(batch,7),generator=generator,device='cuda'),
            'atom_mask':torch.tensor([[1,1,1,0,0],[1,1,1,1,1]]*(batch//2),device='cuda',dtype=torch.bool),
            'residue_mask':torch.tensor([[1,1,1,1,0,0,0],[1,1,1,1,1,1,1]]*(batch//2),device='cuda',dtype=torch.bool),
            'drug_global':torch.randn(batch,128,generator=generator,device='cuda'),
            'protein_global':torch.randn(batch,128,generator=generator,device='cuda')}


def parent_class():
    spec=importlib.util.spec_from_file_location('conditional_parent_pair_interaction',PARENT/'pair_interaction.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.PairResidual


def train_step(model,optimizer,b,target):
    model.train();optimizer.zero_grad(set_to_none=True)
    loss=(model(b)-target).square().mean();loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()),float(norm)


def model_checks():
    b=synthetic();Parent=parent_class()
    seed_all(42);parent=Parent('pair_graph',selection='coverage',coverage_budget=32,attention_mode='none').cuda()
    models={}
    for variant in ['baseline','atom_conditioned','residue_conditioned','bidirectional']:
        seed_all(42);models[variant]=PairResidual(variant).cuda()
    baseline=models['baseline']
    assert parent.state_dict().keys()==baseline.state_dict().keys()
    assert all(torch.equal(parent.state_dict()[k],baseline.state_dict()[k]) for k in parent.state_dict())
    for variant in ['atom_conditioned','residue_conditioned','bidirectional']:
        common={k:v for k,v in models[variant].state_dict().items() if not k.startswith('cross.')}
        assert baseline.state_dict().keys()==common.keys()
        assert all(torch.equal(baseline.state_dict()[k],common[k]) for k in common)
    for model in [parent,*models.values()]: model.eval()
    assert all(torch.equal(parent(b),model(b)) for model in models.values())
    with torch.no_grad():
        parent.output.weight.normal_(0,.1)
        for model in models.values(): model.output.weight.copy_(parent.output.weight)
    assert all(torch.equal(parent(b),model(b)) for model in models.values())

    # Padding and valid-node reindexing must not affect the bidirectional result.
    bidir=models['bidirectional']
    with torch.no_grad():
        bidir.cross.atom_gamma.fill_(.4);bidir.cross.residue_gamma.fill_(.4)
    bidir.eval();clean=bidir(b)
    padding={k:v.clone() for k,v in b.items()}
    padding['atoms'][~b['atom_mask']]=1000;padding['residues'][~b['residue_mask']]=-1000
    padding['aa'][~b['residue_mask']]=0
    assert torch.allclose(bidir(padding),clean,atol=1e-6,rtol=0)
    perm={k:v.clone() for k,v in b.items()};ap=torch.tensor([2,0,1,4,3],device='cuda');rp=torch.tensor([3,0,5,1,6,2,4],device='cuda')
    for key in ['atoms','atom_mask']:perm[key]=perm[key][:,ap]
    for key in ['residues','residue_mask','aa']:perm[key]=perm[key][:,rp]
    assert torch.allclose(bidir(perm),clean,atol=1e-6,rtol=0)
    other={k:v.clone() for k,v in b.items()}
    for key in ['atoms','residues','drug_global','protein_global']:other[key][1]*=100
    assert torch.allclose(bidir(other)[0],clean[0],atol=1e-6,rtol=0)
    assert float((clean-bidir(b,cross_override='off')).abs().max())>1e-8
    assert float((clean-bidir(b,cross_override='shuffle_opposite')).abs().max())>1e-8
    _,info=bidir(b,diagnostics=True)
    assert info['cross_atom_update_rms'].mean()>0 and info['cross_residue_update_rms'].mean()>0

    # At zero gate the first gradient opens each direction; the second reaches Q/K/V.
    seed_all(81);gradient_model=PairResidual('bidirectional').cuda()
    with torch.no_grad():gradient_model.output.weight.normal_(0,.1)
    optimizer=torch.optim.AdamW(gradient_model.parameters(),lr=3e-4)
    target=torch.tensor([.1,-.2,.05,-.15],device='cuda')
    gradient_model.train();optimizer.zero_grad(set_to_none=True)
    loss=(gradient_model(b)-target).square().mean();loss.backward()
    assert gradient_model.cross.atom_gamma.grad.abs()>0
    assert gradient_model.cross.residue_gamma.grad.abs()>0
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    loss=(gradient_model(b)-target).square().mean();loss.backward()
    for parameter in [gradient_model.cross.atom_q.weight,gradient_model.cross.residue_k.weight,
                      gradient_model.cross.residue_q.weight,gradient_model.cross.atom_k.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
    optimizer.step()

    saved_model=copy.deepcopy(gradient_model.state_dict());saved_opt=copy.deepcopy(optimizer.state_dict())
    state=rng_state();expected,_=train_step(gradient_model,optimizer,b,target)
    resumed=PairResidual('bidirectional').cuda();resumed.load_state_dict(saved_model)
    resumed_optimizer=torch.optim.AdamW(resumed.parameters(),lr=3e-4);resumed_optimizer.load_state_dict(saved_opt)
    restore_rng(state);actual,_=train_step(resumed,resumed_optimizer,b,target)
    assert expected==actual
    resume_error=max(float((x-y).abs().max()) for x,y in zip(gradient_model.parameters(),resumed.parameters()))
    assert resume_error<1e-7,resume_error
    parameters={name:sum(p.numel() for p in model.parameters()) for name,model in models.items()}
    return {'parent_baseline_state_exact':True,'zero_gate_all_variants_output_exact':True,
            'padding_permutation_batch_isolation':True,'both_directions_change_output':True,
            'opposite_entity_shuffle_changes_output':True,'two_step_qkv_gradient_nonzero':True,
            'resume_loss_exact':True,'resume_parameter_max_error':resume_error,
            'parameters':parameters,'cross_parameter_increment':parameters['bidirectional']-parameters['baseline']}


def main():
    torch.set_num_threads(4);checks=model_checks();assert checks['cross_parameter_increment']==8578
    cache=torch.load(CACHE_SOURCE/'fold_1.pt',map_location='cpu',weights_only=False);store=FeatureStore(cache)
    train=cache['split_drugs']['train']
    proteins=np.array(sorted(range(len(cache['proteins'])),key=lambda x:store.protein_length[x],reverse=True)[:16])
    drugs=np.array([train[i%len(train)] for i in range(16)]);batch=store.batch(drugs,proteins)
    model=PairResidual('bidirectional',cross_max_scale=load_protocol()['cross_attention']['max_residual_scale']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4);seed_all(999);torch.cuda.reset_peak_memory_stats();begun=time.time()
    target=torch.as_tensor(cache['labels'][drugs,proteins]-cache['base'][drugs,proteins],device='cuda',dtype=torch.float32)
    for _ in range(5):train_step(model,optimizer,batch,target)
    torch.cuda.synchronize()
    assert model.cross.atom_q.weight.grad.abs().sum()>0 and model.cross.residue_q.weight.grad.abs().sum()>0
    r1=R1(cache);val=cache['split_drugs']['val'];original=r1.correct(cache['base'],val)[0]
    assert np.array_equal(original,r1.correct(cache['base']+0*np.ones_like(cache['base']),val)[0])
    shifted=cache['base'].copy();shifted[train]+=.1
    assert not np.array_equal(r1.correct(shifted,val)[0],original)
    result={'passed':True,'checks':checks,'stage':'after frozen Rank16 node features and before pair selection',
            'sources':{'atoms':'drug_3d_encoder(..., return_node=True)[node_feat], 128D',
                       'residues':'protein_3d_encoder(..., return_node=True)[node_feat], 128D + 8D residue type'},
            'real_max_length_batch':{'atoms_shape':list(batch['atoms'].shape),'residues_shape':list(batch['residues'].shape),
                'five_steps_seconds':time.time()-begun,'peak_cuda_mib':torch.cuda.max_memory_allocated()/1024**2},
            'disabled_r1_exact':True,'r1_rebuild_responds_to_combined_prediction':True,'timestamp':time.time()}
    dump(OUT/'preflight.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
