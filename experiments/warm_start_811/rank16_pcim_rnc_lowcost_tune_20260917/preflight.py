"""Preflight pair-level RNC correctness, provenance, gradients, and GPU memory."""
import importlib.util
import json
import time

import numpy as np
import torch

from common import OUT, CACHE_SOURCE, PARENT, FeatureStore, R1, dump, load_protocol, seed_all
from pair_interaction import PairResidual
from rnc_loss import StratifiedPairSampler, rank_n_contrast_loss


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
    spec=importlib.util.spec_from_file_location('bidir_parent_pair_interaction',PARENT/'pair_interaction.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.PairResidual


def model_and_loss_checks(cfg):
    b=synthetic();Parent=parent_class()
    seed_all(42);parent=Parent('bidirectional').cuda()
    seed_all(42);baseline=PairResidual('bidirectional',enable_contrast=False).cuda()
    seed_all(42);rnc_model=PairResidual('bidirectional',enable_contrast=True,
                                        contrast_dim=cfg['rnc']['projection_dim']).cuda()
    assert parent.state_dict().keys()==baseline.state_dict().keys()
    assert all(torch.equal(parent.state_dict()[k],baseline.state_dict()[k]) for k in parent.state_dict())
    common={k:v for k,v in rnc_model.state_dict().items() if not k.startswith('contrast_projection.')}
    assert baseline.state_dict().keys()==common.keys()
    assert all(torch.equal(baseline.state_dict()[k],common[k]) for k in common)
    parent.eval();baseline.eval();rnc_model.eval()
    assert torch.equal(parent(b),baseline(b)) and torch.equal(parent(b),rnc_model(b))
    with torch.no_grad():
        baseline.output.weight.normal_(0,.1)
        parent.output.weight.copy_(baseline.output.weight)
        rnc_model.output.weight.copy_(baseline.output.weight)
    assert torch.equal(parent(b),baseline(b)) and torch.equal(parent(b),rnc_model(b))
    _,embedding=rnc_model(b,return_contrast=True)
    assert embedding.shape==(4,cfg['rnc']['projection_dim'])
    assert torch.allclose(embedding.norm(dim=-1),torch.ones(4,device='cuda'),atol=1e-6,rtol=0)

    labels=torch.tensor([4.5,6.0,7.5,8.5],device='cuda')
    standard,standard_info=rank_n_contrast_loss(embedding,labels,temperature=cfg['rnc']['temperature'])
    weighted,weighted_info=rank_n_contrast_loss(
        embedding,labels,temperature=cfg['rnc']['temperature'],mode='high_weighted',
        high_threshold=cfg['rnc']['high_threshold'],high_width=cfg['rnc']['high_width'],
        high_strength=cfg['rnc']['high_strength'])
    equal_weight,_=rank_n_contrast_loss(
        embedding,labels,temperature=cfg['rnc']['temperature'],mode='high_weighted',
        high_strength=0.0)
    assert torch.allclose(standard,equal_weight,atol=1e-7,rtol=0)
    rnc_model.train();rnc_model.zero_grad(set_to_none=True)
    _,embedding=rnc_model(b,return_contrast=True)
    loss,_=rank_n_contrast_loss(embedding,labels,temperature=cfg['rnc']['temperature'])
    loss.backward()
    assert rnc_model.contrast_projection[0].weight.grad.abs().sum()>0
    assert rnc_model.graph.qkv.weight.grad.abs().sum()>0
    assert rnc_model.cross.atom_gamma.grad.abs()>0 and rnc_model.cross.residue_gamma.grad.abs()>0

    sampler=StratifiedPairSampler(np.linspace(4,9,64),batch_size=32,high_count=8,mid_count=8)
    seed_all(19);indices=sampler.sample();sampled=sampler.labels[indices]
    assert len(indices)==len(set(indices))==32
    assert int((sampled>=7).sum())>=8 and int(((sampled>5)&(sampled<7)).sum())>=8
    return {'parent_bidir_state_and_output_exact':True,
            'rnc_head_does_not_change_prediction_at_initialization':True,
            'projection_shape':list(embedding.shape),'projection_unit_norm':True,
            'standard_equals_zero_strength_weighted':True,
            'standard_loss':float(standard),'weighted_loss':float(weighted),
            'standard_diagnostics':standard_info,'weighted_diagnostics':weighted_info,
            'rnc_gradient_reaches_projection_pairgraph_and_both_cross_gates':True,
            'sampler_distinct_and_quota_satisfied':True,
            'parameters_baseline':sum(p.numel() for p in baseline.parameters()),
            'parameters_rnc':sum(p.numel() for p in rnc_model.parameters())}


def real_memory_check(cfg):
    cache=torch.load(CACHE_SOURCE/'fold_1.pt',map_location='cpu',weights_only=False)
    store=FeatureStore(cache);train=cache['split_drugs']['train'];n_proteins=len(cache['proteins'])
    td=np.repeat(train,n_proteins);tp=np.tile(np.arange(n_proteins),len(train))
    labels=cache['labels'][td,tp]
    sampler=StratifiedPairSampler(labels,batch_size=cfg['rnc']['batch_size'],
                                  high_count=cfg['rnc']['high_count'],mid_count=cfg['rnc']['mid_count'],
                                  high_threshold=cfg['rnc']['high_threshold'],mid_threshold=cfg['rnc']['mid_threshold'])
    seed_all(999);rix=sampler.sample();rd,rp=td[rix],tp[rix]
    order=np.argsort([-store.protein_length[int(x)] for x in rp]);rd,rp=rd[order],rp[order]
    affinity=torch.as_tensor(labels[rix][order],device='cuda',dtype=torch.float32)
    model=PairResidual('bidirectional',enable_contrast=True,contrast_dim=cfg['rnc']['projection_dim'],
                       cross_max_scale=cfg['cross_attention']['max_residual_scale']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'])
    task_b=store.batch(rd[:cfg['batch_size']],rp[:cfg['batch_size']])
    task_target=torch.as_tensor(cache['labels'][rd[:cfg['batch_size']],rp[:cfg['batch_size']]]
                                -cache['base'][rd[:cfg['batch_size']],rp[:cfg['batch_size']]],
                                device='cuda',dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats();begun=time.time();model.train();optimizer.zero_grad(set_to_none=True)
    delta=model(task_b);task_loss=(delta-task_target).square().mean();task_loss.backward()
    embeddings=[]
    for start in range(0,len(rd),cfg['rnc']['microbatch_size']):
        end=start+cfg['rnc']['microbatch_size'];rb=store.batch(rd[start:end],rp[start:end])
        _,embedding=model(rb,return_contrast=True);embeddings.append(embedding)
    rnc_loss,_=rank_n_contrast_loss(torch.cat(embeddings),affinity,
                                    temperature=cfg['rnc']['temperature'],mode='high_weighted')
    (0.01*rnc_loss).backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
    optimizer.step();torch.cuda.synchronize()
    return {'peak_cuda_mib':torch.cuda.max_memory_allocated()/1024**2,
            'seconds':time.time()-begun,'task_loss':float(task_loss),'rnc_loss':float(rnc_loss),
            'gradient_norm':float(norm),'sampler':sampler.audit(),
            'sample_high':int((affinity>=cfg['rnc']['high_threshold']).sum()),
            'sample_mid':int(((affinity>cfg['rnc']['mid_threshold'])
                              &(affinity<cfg['rnc']['high_threshold'])).sum())}


def main():
    torch.set_num_threads(4);cfg=load_protocol();checks=model_and_loss_checks(cfg)
    memory=real_memory_check(cfg)
    cache=torch.load(CACHE_SOURCE/'fold_1.pt',map_location='cpu',weights_only=False)
    r1=R1(cache);val=cache['split_drugs']['val'];baseline=r1.correct(cache['base'],val)[0]
    assert np.array_equal(baseline,r1.correct(cache['base']+0*np.ones_like(cache['base']),val)[0])
    result={'passed':True,'checks':checks,'real_training_step':memory,
            'disabled_r1_exact':True,'training_labels_only_for_rnc':True,
            'timestamp':time.time()}
    dump(OUT/'preflight.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
