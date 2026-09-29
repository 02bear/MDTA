import json
from types import SimpleNamespace
import torch
import train_p13d_bilinear as trial

torch.set_num_threads(4)
torch.manual_seed(42)
head = trial.AdditiveBilinearHead().eval()
d, p = torch.randn(4, 128), torch.randn(5, 128)
x = torch.cat([d[:, None].expand(-1, 5, -1), p[None].expand(4, -1, -1)], -1)
y = head(x).squeeze(-1)
mixed = y - y.mean(0, keepdim=True) - y.mean(1, keepdim=True) + y.mean()
assert mixed.square().mean() > 1e-8
y.square().mean().backward()
for name in ('drug_projection', 'protein_projection'):
    g = getattr(head, name).weight.grad
    assert torch.isfinite(g).all() and g.norm() > 0
with torch.no_grad():
    head.drug_projection.weight.zero_()
    a = head(x).squeeze(-1)
    assert (a - a.mean(0, keepdim=True) - a.mean(1, keepdim=True) + a.mean()).abs().max() < 1e-6

root = trial.PROJECT / 'outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/baseline'
for fold in (1, 3):
    ck = torch.load(root / f'fold_{fold}/best_model.pt', map_location='cpu', weights_only=False)
    args = SimpleNamespace(**ck['args'])
    trial.base.set_seed(42)
    original = trial._build(args, torch.device('cpu'))
    trial.base.set_seed(42)
    model = trial.build_model(args, torch.device('cpu'))
    for k, v in original.state_dict().items():
        if not k.startswith('decoder.'):
            assert torch.equal(v, model.state_dict()[k]), k
    ds, tr, val, train_loader, _ = trial.base.build_dataloaders(args)
    assert set(ds.df.iloc[tr.indices].drug_id).isdisjoint(ds.df.iloc[val.indices].drug_id)
    del original, ck
    model = model.cuda().train()
    batch = trial.base.move_batch_to_device(next(iter(train_loader)), torch.device('cuda'))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pred = model(batch)
    assert pred.shape == batch['label'].shape
    loss = torch.nn.functional.mse_loss(pred, batch['label'])
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    assert model.decoder.drug_projection.weight.grad.norm() > 0
    assert model.decoder.protein_projection.weight.grad.norm() > 0
    opt.step()
    from torch.utils.data import DataLoader, Subset
    vf = ds.df.iloc[val.indices]
    selected = vf[vf.drug_id.isin(vf.drug_id.unique()[:2]) & vf.protein_id.isin(vf.protein_id.unique()[:4])].index.tolist()
    check_loader = DataLoader(Subset(ds, selected), batch_size=8, collate_fn=trial.base.mdta_collate_fn_p13d)
    metrics = trial.evaluate(model, check_loader, torch.nn.MSELoss(), torch.device('cuda'))
    assert len(trial._last_validation['y_pred']) == 8
    assert metrics['pred_interaction_rms'] > 0
    print(json.dumps({'fold': fold, 'smoke_passed': True, 'loss': float(loss), 'batch_size': len(pred), 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20}), flush=True)
    del model, opt, batch, pred, loss, ds, train_loader, check_loader
    torch.cuda.empty_cache()
print('ALL_SMOKE_TESTS_PASSED', flush=True)
