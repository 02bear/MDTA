"""Real-data preflight for one unfreezing branch on its assigned GPU."""
import argparse
import json
import time

import numpy as np
import torch

from common import OUT, R1, dump, load_protocol, seed_all, verify_manifest
from end_to_end import EndToEndPairRNC, FoldRuntime, adapter_checkpoint_for_fold
from rnc_loss import StratifiedPairSampler, rank_n_contrast_loss


def grad_sum(module):
    return float(sum((p.grad.abs().sum() for p in module.parameters() if p.grad is not None),
                     torch.zeros((), device='cuda')))


def representation_audit(runtime, model):
    rows = runtime.train_indices[:4]
    raw = runtime.collate_indices(rows)
    batch = runtime.move(raw)
    model.eval()
    with torch.no_grad():
        d3 = model.backbone.drug_3d_encoder(batch['drug_3d'], return_node=True)
        p3 = model.backbone.protein_3d_encoder(batch['protein_3d'], return_node=True)
        dg = model.backbone.drug_fusion([model.backbone.drug_1d_encoder(batch['drug_1d']), d3['graph_feat']])
        pg = model.backbone.protein_fusion([model.backbone.protein_1d_encoder(batch['protein_1d']), p3['graph_feat']])
        base = model.backbone.decoder(torch.cat([dg, pg], -1)).flatten()
    atom_error = residue_error = global_error = base_error = 0.0
    for i, row in enumerate(rows):
        di, pi = runtime.row_to_drug[row], runtime.row_to_protein[row]
        atom = d3['node_feat'][d3['batch'] == i].cpu()
        residue = p3['node_feat'][p3['batch'] == i].cpu()
        atom_error = max(atom_error, float((atom-runtime.cache['atoms'][di]).abs().max()))
        residue_error = max(residue_error, float((residue-runtime.cache['residues'][pi]).abs().max()))
        global_error = max(global_error, float((dg[i].cpu()-runtime.cache['drug_global'][di]).abs().max()),
                           float((pg[i].cpu()-runtime.cache['protein_global'][pi]).abs().max()))
        base_error = max(base_error, abs(float(base[i])-float(runtime.cache['base'][di, pi])))
    assert atom_error < 1e-5 and residue_error < 1e-4 and global_error < 1e-4 and base_error < 1e-4
    return {'atom_max_error': atom_error, 'residue_max_error': residue_error,
            'global_max_error': global_error, 'base_prediction_max_error': base_error}


def phase_gradient_audit(runtime, model, variant):
    rows = runtime.train_indices[:8]
    raw = runtime.collate_indices(rows)
    labels = runtime.dataset.df.iloc[rows].label.to_numpy(dtype=np.float32)
    results = {}
    for epoch in [1, 3, 6]:
        model.zero_grad(set_to_none=True)
        phase = model.configure_phase(epoch); model.set_training_mode()
        batch = runtime.move(raw)
        base, delta, embedding = model(batch, return_contrast=True)
        target = batch['label'].flatten()
        rnc, _ = rank_n_contrast_loss(embedding, torch.as_tensor(labels, device='cuda'),
                                      temperature=2.0, mode='standard')
        ((base+delta-target).square().mean()+0.01*rnc).backward()
        head, last, one_d, early = model._stage_modules()
        audit = {'phase': phase, 'interaction_grad': grad_sum(model.interaction),
                 'head_grad': sum(grad_sum(x) for x in head),
                 'last_grad': sum(grad_sum(x) for x in last),
                 'one_d_grad': sum(grad_sum(x) for x in one_d),
                 'early_grad': sum(grad_sum(x) for x in early)}
        assert audit['interaction_grad'] > 0
        if epoch == 1:
            assert audit['head_grad'] == audit['last_grad'] == audit['one_d_grad'] == audit['early_grad'] == 0
        if epoch == 3:
            assert audit['head_grad'] > 0 and audit['last_grad'] > 0 and audit['one_d_grad'] > 0
            assert audit['early_grad'] == 0
        if epoch == 6 and variant == 'unfreeze_full':
            assert audit['early_grad'] > 0
        if epoch == 6 and variant == 'unfreeze_last':
            assert audit['early_grad'] == 0
        results[str(epoch)] = audit
    return results


def memory_step(runtime, model, cfg, variant):
    model.zero_grad(set_to_none=True)
    model.configure_phase(6); model.set_training_mode()
    optimizer = torch.optim.AdamW(model.optimizer_groups(cfg), weight_decay=cfg['weight_decay'])
    labels = runtime.dataset.df.iloc[runtime.train_indices].label.to_numpy(dtype=np.float64)
    sampler = StratifiedPairSampler(labels, batch_size=cfg['rnc']['batch_size'],
                                    high_count=cfg['rnc']['high_count'], mid_count=cfg['rnc']['mid_count'],
                                    high_threshold=cfg['rnc']['high_threshold'], mid_threshold=cfg['rnc']['mid_threshold'])
    task_rows = runtime.train_indices[:cfg['batch_size']]
    positions = sampler.sample(); rnc_rows = runtime.train_indices[positions]
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); began = time.time()
    batch = runtime.move(runtime.collate_indices(task_rows))
    base, delta = model(batch); target = batch['label'].flatten()
    task_loss = (base+delta-target).square().mean()+cfg['residual_l2']*delta.square().mean()
    task_loss.backward()
    embeddings = []
    for start in range(0, len(rnc_rows), cfg['rnc']['microbatch_size']):
        rb = runtime.move(runtime.collate_indices(rnc_rows[start:start+cfg['rnc']['microbatch_size']]))
        _, _, embedding = model(rb, return_contrast=True); embeddings.append(embedding)
    affinity = torch.as_tensor(labels[positions], dtype=torch.float32, device='cuda')
    rnc_loss, _ = rank_n_contrast_loss(torch.cat(embeddings), affinity,
                                       temperature=cfg['rnc']['temperature'], mode='standard')
    (cfg['rnc_weight']*cfg['rnc']['interval']*rnc_loss).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
    optimizer.step(); torch.cuda.synchronize()
    return {'variant': variant, 'peak_cuda_mib': torch.cuda.max_memory_allocated()/1024**2,
            'seconds': time.time()-began, 'task_loss': float(task_loss.detach()),
            'rnc_loss': float(rnc_loss.detach()), 'gradient_norm': float(norm),
            'trainable': model.trainable_audit(), 'sampler': sampler.audit()}


def main(variant):
    verify_manifest(); cfg = load_protocol(); seed_all(42)
    runtime = FoldRuntime(1)
    adapter, _ = adapter_checkpoint_for_fold(1)
    model = EndToEndPairRNC(runtime, variant, adapter).cuda()
    representation = representation_audit(runtime, model)
    gradients = phase_gradient_audit(runtime, model, variant)
    memory = memory_step(runtime, model, cfg, variant)
    val = runtime.cache['split_drugs']['val']; r1 = R1(runtime.cache)
    original = r1.correct(runtime.cache['base'], val)[0]
    assert np.array_equal(original, r1.correct(runtime.cache['base']+0, val)[0])
    result = {'passed': True, 'variant': variant, 'representation': representation,
              'phase_gradients': gradients, 'real_compound_step': memory,
              'original_rank16_r1_fallback_exact': True,
              'adapter_checkpoint': str(adapter), 'timestamp': time.time()}
    dump(OUT/f'preflight_{variant}.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--variant', required=True,
        choices=['unfreeze_last','unfreeze_full'])
    main(parser.parse_args().variant)
