"""Behavioral tests on synthetic and real features; no experiment checkpoint is trained here."""
import copy
import json
import time
import numpy as np
import torch
from common import OUT, FeatureStore, R1, load_protocol, dump, seed_all, rng_state, restore_rng
from pair_interaction import PairResidual


def synthetic():
    device='cuda'
    generator=torch.Generator(device=device).manual_seed(71)
    return {'atoms':torch.randn(2,5,128,generator=generator,device=device),
            'residues':torch.randn(2,7,128,generator=generator,device=device),
            'aa':torch.randint(0,21,(2,7),generator=generator,device=device),
            'atom_mask':torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],device=device,dtype=torch.bool),
            'residue_mask':torch.tensor([[1,1,1,1,0,0,0],[1,1,1,1,1,1,1]],device=device,dtype=torch.bool),
            'drug_global':torch.randn(2,128,generator=generator,device=device),
            'protein_global':torch.randn(2,128,generator=generator,device=device)}


def test_synthetic():
    checks={};parameters={}
    for variant in ['pair_graph','pair_pool','global_mlp']:
        seed_all(42)
        m=PairResidual(variant).cuda()
        b=synthetic()
        m.eval()
        assert torch.equal(m(b),torch.zeros(2,device='cuda'))
        with torch.no_grad():m.output.weight.normal_(0,.1)
        clean=m(b)
        assert torch.isfinite(clean).all() and (clean.abs()<=.5).all()
        other={k:v.clone() for k,v in b.items()}
        other['drug_global'][1]*=100
        other['protein_global'][1]*=100
        other['atoms'][1]*=100
        other['residues'][1]*=100
        assert torch.allclose(m(other)[0],clean[0],atol=1e-6,rtol=0)
        if variant!='global_mlp':
            padding={k:v.clone() for k,v in b.items()}
            padding['atoms'][~b['atom_mask']]=1000
            padding['residues'][~b['residue_mask']]=-1000
            padding['aa'][~b['residue_mask']]=0
            assert torch.allclose(m(padding),clean,atol=1e-6,rtol=0)
            # Valid reindexing must preserve prediction; this is not a destructive shuffle control.
            perm={k:v.clone() for k,v in b.items()}
            ap=torch.tensor([2,0,1,4,3],device='cuda')
            pp=torch.tensor([3,0,5,1,6,2,4],device='cuda')
            for key in ['atoms','atom_mask']:perm[key]=perm[key][:,ap]
            for key in ['residues','residue_mask','aa']:perm[key]=perm[key][:,pp]
            assert torch.allclose(m(perm),clean,atol=1e-6,rtol=0)
            single={k:v[:1].clone() for k,v in b.items()}
            single['atoms']=single['atoms'][:,:3];single['atom_mask']=single['atom_mask'][:,:3]
            for key in ['residues','aa','residue_mask']:single[key]=single[key][:,:4]
            assert torch.allclose(m(single)[0],clean[0],atol=1e-6,rtol=0)
        # A nonzero readout must transmit gradients into both projections and condition.
        m.train()
        opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
        opt.zero_grad(); loss=(m(b)-torch.tensor([.1,-.2],device='cuda')).square().mean();loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        if variant!='global_mlp':
            for p in [m.atom_projection[0].weight,m.protein_projection[0].weight,m.condition[0].weight]:
                assert p.grad is not None and p.grad.abs().sum()>0
        opt.step()
        # Serialize optimizer + RNG, and compare the next actual training step.
        saved_model=copy.deepcopy(m.state_dict());saved_opt=copy.deepcopy(opt.state_dict());state=rng_state()
        def step(mm,oo):
            oo.zero_grad(); l=(mm(b)-.2).square().mean();l.backward();oo.step();return float(l.detach())
        expected=step(m,opt)
        resumed=PairResidual(variant).cuda();resumed.load_state_dict(saved_model);resumed.train()
        resumed_opt=torch.optim.AdamW(resumed.parameters(),lr=3e-4);resumed_opt.load_state_dict(saved_opt)
        restore_rng(state)
        actual=step(resumed,resumed_opt)
        assert expected==actual
        assert all(torch.equal(x,y) for x,y in zip(m.parameters(),resumed.parameters()))
        parameters[variant]=sum(p.numel() for p in m.parameters())
        checks[variant]=dict(zero_init=True,batch_isolation=True,padding_and_permutation=variant!='global_mlp',
                             finite_gradients=True,optimizer_rng_resume_exact=True)
    return checks,parameters


def main():
    torch.set_num_threads(4)
    checks,params=test_synthetic()
    cache=torch.load(OUT/'cache/fold_1.pt',map_location='cpu',weights_only=False)
    store=FeatureStore(cache)
    cfg=load_protocol()
    model=PairResidual('pair_graph').cuda()
    train=cache['split_drugs']['train']
    p=np.array(sorted(range(len(cache['proteins'])),key=lambda x:store.protein_length[x],reverse=True)[:16])
    d=np.array([train[i%len(train)] for i in range(16)])
    b=store.batch(d,p)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    seed_all(999)
    torch.cuda.reset_peak_memory_stats()
    begun=time.time()
    for i in range(5):
        model.train();optimizer.zero_grad()
        delta=model(b)
        target=torch.as_tensor(cache['labels'][d,p]-cache['base'][d,p],device='cuda',dtype=torch.float32)
        loss=(delta-target).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
    torch.cuda.synchronize()
    assert model.atom_projection[0].weight.grad.abs().sum()>0
    assert model.protein_projection[0].weight.grad.abs().sum()>0
    r1=R1(cache)
    original=r1.correct(cache['base'],cache['split_drugs']['val'])[0]
    disabled=r1.correct(cache['base']+0*np.ones_like(cache['base']),cache['split_drugs']['val'])[0]
    assert np.array_equal(original,disabled)
    # Newly computed training residuals must respond to training prediction changes.
    shifted=cache['base'].copy();shifted[train]+=.1
    changed=r1.correct(shifted,cache['split_drugs']['val'])[0]
    assert not np.array_equal(changed,original)
    result={'passed':True,'checks':checks,'parameters':params,'real_max_length_batch':{
        'atoms_shape':list(b['atoms'].shape),'residues_shape':list(b['residues'].shape),
        'five_steps_seconds':time.time()-begun,'peak_cuda_mib':torch.cuda.max_memory_allocated()/1024**2},
        'disabled_r1_exact':True,'r1_rebuild_responds_to_combined_prediction':True,
        'timestamp':time.time()}
    dump(OUT/'preflight.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
