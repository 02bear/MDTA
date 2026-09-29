"""Integration preflight: paired heads, sampler provenance, actual cached data."""
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from common import OUTPUT, dump, assert_sources
from model_frozen_ot import FrozenOTHead
from train_frozen_ot import mappings, parameters_hash, make_loader, CachedPairs
from train_p13d_earlystop import set_seed


def verify():
    torch.set_num_threads(1)
    root=OUTPUT/'fold_1'
    cache=torch.load(root/'cache/features.pt',map_location='cpu',weights_only=False)
    assert_sources(cache)
    ids=set(cache['entities']['drug']['train_ids'])
    assert ids==set(cache['pairs']['train']['drug_id'])
    assert not ids&set(cache['pairs']['val']['drug_id'])
    models=[];hashes=[]
    ds=CachedPairs(cache,'train')
    batch=next(iter(make_loader(ds,42,16,False)))
    for mode in ['F0','F1','F2','F3','F4']:
        set_seed(42)
        d,p=mappings(root,mode,128)
        model=FrozenOTHead(d,p)
        hashes.append(parameters_hash(model))
        model.eval()
        prediction=model(batch)
        assert prediction.shape==(16,1) and torch.isfinite(prediction).all()
        loss=torch.nn.functional.mse_loss(prediction,batch['label'])
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(not b.requires_grad for b in model.buffers())
        models.append(model)
    assert len(set(hashes))==1
    # F0 identity buffer is exactly the original fusion/decoder functional path.
    m=models[0]
    with torch.no_grad():
        direct=m.decoder(torch.cat([m.drug_fusion([batch['drug_1d'],batch['drug_3d']]),
                                    m.protein_fusion([batch['protein_1d'],batch['protein_3d']])],-1))
        torch.testing.assert_close(m(batch),direct,atol=0,rtol=0)
    class IndexDataset(Dataset):
        def __len__(self):return len(ds)
        def __getitem__(self,i):return i
    actual_order=torch.cat(list(make_loader(IndexDataset(),42,16,True))).numpy()
    probe=make_loader(IndexDataset(),42,16,True)
    torch.empty((),dtype=torch.int64).random_(generator=probe.generator)
    probe_order=np.array([i for b in probe.batch_sampler for i in b],dtype=np.int64)
    np.testing.assert_array_equal(actual_order,probe_order)
    diag={}
    for name in ['drug','drug_shuffle']:
        obj=torch.load(root/'ot'/(name+'_ot_matrix.pt'),weights_only=False)
        assert set(obj['metadata']['train_entity_ids'])==ids
        diag[name]=obj['metadata']['diagnostics']
    t1=torch.load(root/'ot/drug_ot_matrix.pt',weights_only=False)['T']
    t2=torch.load(root/'ot/drug_shuffle_ot_matrix.pt',weights_only=False)['T']
    bootstrap=json.loads((root/'ot/bootstrap.json').read_text())
    report=dict(passed=True,paired_initialization_hash=hashes[0],
        first_epoch_sampler_order_sha256=hashlib.sha256(actual_order.astype(np.int64).tobytes()).hexdigest(),
        original_fusion_identity_exact=True,all_modes_finite_forward_backward=True,
        unique_train_drugs=len(ids),validation_drugs_excluded_from_OT=True,
        diagnostics=diag,relative_shuffled_plan_difference=float(torch.linalg.norm(t1-t2)/torch.linalg.norm(t1)),
        bootstrap_converged=sum(t['converged'] for t in bootstrap['trials']),
        bootstrap_repeats=bootstrap['repeats'])
    dump(root/'preflight.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':verify()
