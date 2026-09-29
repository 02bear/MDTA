"""Mechanistic selection tests and legacy first-epoch replay; validation only."""
import importlib.util
import json
import time
import numpy as np
import torch
from common import PARENT, PARENT_OUT, OUT, seed_all, dump, load_protocol
from pair_interaction import PairResidual, select_pairs

def test_selection():
    checks=[]
    for atoms,residues,padded_a,padded_r in [(1,1,3,5),(5,7,40,80),(25,10,40,20),(80,7,80,10)]:
        values=torch.arange(padded_a*padded_r,device='cuda',dtype=torch.float32)
        scores=(-values).reshape(1,padded_a,padded_r)
        scores[:,atoms:]=-torch.inf
        scores[:,:,residues:]=-torch.inf
        count=min(64,padded_a*padded_r)
        sv,idx=select_pairs(scores,count,'coverage')
        good=torch.isfinite(sv[0]);chosen=idx[0][good]
        assert len(chosen)==min(64,atoms*residues)
        assert chosen.unique().numel()==chosen.numel()
        assert ((chosen//padded_r)<atoms).all() and ((chosen%padded_r)<residues).all()
        assert (chosen//padded_r).unique().numel()>=min(32,atoms,count)
        # The exact set is reserved per-atom winners + highest unreserved scores.
        winners=sorted([(float(scores[0,a].max()),a*padded_r+int(scores[0,a].argmax())) for a in range(atoms)],reverse=True)[:min(32,atoms,count)]
        reserved={x[1] for x in winners}
        rest=sorted([(float(scores[0,a,r]),a*padded_r+r) for a in range(atoms) for r in range(residues) if a*padded_r+r not in reserved],reverse=True)
        expected=reserved|{x[1] for x in rest[:min(64,atoms*residues)-len(reserved)]}
        assert set(chosen.tolist())==expected
        checks.append({'atoms':atoms,'residues':residues,'valid_pairs':len(chosen),
                       'distinct_atoms':int((chosen//padded_r).unique().numel())})
    x=torch.randn(2,40,10,device='cuda',requires_grad=True)
    v,_=select_pairs(x,64,'coverage');v.sum().backward()
    assert torch.isfinite(x.grad).all() and int((x.grad!=0).sum())==128
    return checks

def replay_first_epoch(cache,store):
    from train_experiment import pairs,infer,select_scale
    from common import R1
    cfg=load_protocol()
    train,val=(cache['split_drugs'][k] for k in ['train','val'])
    td,tp=pairs(train,len(cache['proteins']))
    spec=importlib.util.spec_from_file_location('legacy_pair',PARENT/'pair_interaction.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report={}
    for variant in ['pair_graph','pair_pool']:
        seed_all(42)
        new=PairResidual(variant,selection='global').cuda()
        seed_all(42)
        legacy=module.PairResidual(variant).cuda()
        assert all(torch.equal(v,legacy.state_dict()[k]) for k,v in new.state_dict().items())
        with torch.no_grad(): legacy.output.weight.normal_(0,0.1)
        new.load_state_dict(legacy.state_dict())
        new.train();legacy.train()
        batch=store.batch(td[:16],tp[:16])
        state=torch.cuda.get_rng_state()
        actual=new(batch);actual.square().mean().backward()
        torch.cuda.set_rng_state(state)
        expected=legacy(batch);expected.square().mean().backward()
        assert torch.equal(actual,expected),variant
        for (name,p),(_,q) in zip(new.named_parameters(),legacy.named_parameters()):
            if p.grad is not None:
                assert q.grad is not None and torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-4),(variant,name)
        # Reproduce original initialization and training RNG independently for each variant.
        seed_all(42)
        model=PairResidual(variant,selection='global').cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        model.train();order=np.random.permutation(len(td));losses=[]
        began=time.time()
        for start in range(0,len(order),cfg['batch_size']):
            ix=order[start:start+cfg['batch_size']];d,p=td[ix],tp[ix]
            b=store.batch(d,p)
            target=torch.as_tensor(cache['labels'][d,p]-cache['base'][d,p],device='cuda',dtype=torch.float32)
            opt.zero_grad(set_to_none=True);delta=model(b)
            loss=(delta-target).square().mean()+cfg['residual_l2']*delta.square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()))
        n=len(cache['proteins'])
        delta=np.zeros_like(cache['base'])
        delta[train]=infer(model,store,train,n,cfg['eval_batch_size'])
        delta[val]=infer(model,store,val,n,cfg['eval_batch_size'])
        _,choices=select_scale(cache['base'],delta,R1(cache),val,cache['labels'][val],cfg['lambdas'])
        old=json.loads((PARENT_OUT/f'runs/{variant}/seed_42/fold_1/history.json').read_text())[0]
        loss_error=abs(float(np.mean(losses))-old['train_loss'])
        val_error=max(abs(x['mse']-y['mse']) for x,y in zip(choices,old['validation_candidates']))
        assert np.isfinite(loss_error) and np.isfinite(val_error)
        report[variant]={'initial_weights_exact':True,'same_batch_forward_exact':True,
                         'same_batch_gradients_match':True,'historical_results_reused':False,
                         'fresh_global_controls_required':True,'first_epoch_train_loss_error':loss_error,
                         'first_epoch_validation_mse_max_error':val_error,'seconds':time.time()-began}
        print('LEGACY_REPLAY',variant,json.dumps(report[variant]),flush=True)
    return report
