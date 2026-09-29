"""Two-branch end-to-end unfreezing experiment with standard pair-level RNC."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

from common import (HERE, OUT, PARENT_OUT, PROJECT, R1, dump, load_protocol,
                    metric, restore_rng, rng_state, save, seed_all, sha,
                    verify_manifest)
from end_to_end import (EndToEndPairRNC, FoldRuntime,
                        adapter_checkpoint_for_fold, locked_pair_development)
from rnc_loss import StratifiedPairSampler, rank_n_contrast_loss


@torch.no_grad()
def infer_rows(model, runtime, row_indices, batch_size):
    model.eval()
    bases, deltas = [], []
    for batch in runtime.loader(row_indices, batch_size, shuffle=False):
        moved = runtime.move(batch)
        base, delta = model(moved)
        bases.append(base.detach().cpu())
        deltas.append(delta.detach().cpu())
    return torch.cat(bases).numpy().astype(np.float64), torch.cat(deltas).numpy().astype(np.float64)


def query_values(values, runtime, query_drugs, row_indices):
    position = {int(drug): i for i, drug in enumerate(query_drugs)}
    return np.asarray([values[position[int(runtime.row_to_drug[row])], int(runtime.row_to_protein[row])]
                       for row in row_indices], dtype=np.float64)


def initial_candidates(runtime):
    cache = runtime.cache
    val_drugs = cache['split_drugs']['val']
    val_rows = runtime.val_indices
    labels = runtime.dataset.df.iloc[val_rows].label.to_numpy(dtype=np.float64)
    r1 = R1(cache)
    locked_rank = r1.correct(cache['base'], val_drugs)[0]
    locked_pair_matrix, parent_result = locked_pair_development(runtime)
    locked_pair = r1.correct(locked_pair_matrix, val_drugs)[0]
    candidates = [
        {'source': 'locked_pair_rnc', 'lambda': float(parent_result['selection']['lambda']),
         'mse': float(np.mean((query_values(locked_pair, runtime, val_drugs, val_rows)-labels)**2)),
         'epoch': 0},
        {'source': 'locked_rank16', 'lambda': 0.0,
         'mse': float(np.mean((query_values(locked_rank, runtime, val_drugs, val_rows)-labels)**2)),
         'epoch': 0},
    ]
    return min(candidates, key=lambda x: x['mse']), candidates


def select_dynamic(runtime, train_base, train_delta, val_base, val_delta, epoch, lambdas):
    cache = runtime.cache
    train_rows, val_rows = runtime.train_indices, runtime.val_indices
    val_drugs = cache['split_drugs']['val']
    labels = runtime.dataset.df.iloc[val_rows].label.to_numpy(dtype=np.float64)
    r1 = R1(cache)
    initial, locked = initial_candidates(runtime)
    candidates = list(locked)
    base_matrix = runtime.fill_matrix(cache['base'], train_rows, train_base)
    base_matrix = runtime.fill_matrix(base_matrix, val_rows, val_base)
    delta_matrix = np.zeros_like(cache['base'], dtype=np.float64)
    delta_matrix = runtime.fill_matrix(delta_matrix, train_rows, train_delta)
    delta_matrix = runtime.fill_matrix(delta_matrix, val_rows, val_delta)
    for scale in lambdas:
        prediction = base_matrix + float(scale) * delta_matrix
        corrected = r1.correct(prediction, val_drugs)[0]
        values = query_values(corrected, runtime, val_drugs, val_rows)
        candidates.append({'source': 'finetuned', 'lambda': float(scale),
                           'mse': float(np.mean((values-labels)**2)), 'epoch': int(epoch)})
    return min(candidates, key=lambda x: x['mse']), candidates


def train_run(fold, variant, seed=42):
    verify_manifest()
    cfg = load_protocol()
    assert variant in cfg['variants'] and seed == 42
    dest = OUT/f'runs/{variant}/seed_{seed}/fold_{fold}'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'result.json').exists():
        print('RUN_REUSED', variant, fold, flush=True)
        return
    runtime = FoldRuntime(fold)
    adapter_path, parent_result = adapter_checkpoint_for_fold(fold)
    assert sha(adapter_path) == cfg['adapter_initialization'][str(fold)]['sha256']
    seed_all(seed)
    model = EndToEndPairRNC(runtime, variant, adapter_path).cuda()
    optimizer = torch.optim.AdamW(model.optimizer_groups(cfg), weight_decay=cfg['weight_decay'])
    labels = runtime.dataset.df.iloc[runtime.train_indices].label.to_numpy(dtype=np.float64)
    sampler = StratifiedPairSampler(
        labels, batch_size=cfg['rnc']['batch_size'], high_count=cfg['rnc']['high_count'],
        mid_count=cfg['rnc']['mid_count'], high_threshold=cfg['rnc']['high_threshold'],
        mid_threshold=cfg['rnc']['mid_threshold'])
    best, initial = initial_candidates(runtime)
    history = []
    stale = 0
    start_epoch = 1
    if (dest/'latest.pt').exists():
        checkpoint = torch.load(dest/'latest.pt', map_location='cpu', weights_only=False)
        assert checkpoint['protocol'] == cfg and checkpoint['variant'] == variant
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best, history, stale = checkpoint['best'], checkpoint['history'], checkpoint['stale']
        start_epoch = checkpoint['epoch'] + 1
        restore_rng(checkpoint['rng'])
        print('RESUME', variant, fold, start_epoch, flush=True)
    else:
        save(dest/'best.pt', {'model': model.state_dict(), 'selection': best, 'protocol': cfg,
                              'variant': variant, 'seed': seed, 'fold': fold})
    print('TRAIN_START', json.dumps({'variant': variant, 'fold': fold, 'seed': seed,
          'adapter_checkpoint': str(adapter_path), 'adapter_parent_selection': parent_result['selection'],
          'initial_candidates': initial, 'sampler': sampler.audit(),
          'parameters': sum(p.numel() for p in model.parameters())}), flush=True)
    started = time.time()
    for epoch in range(start_epoch, cfg['epochs']+1):
        if stale >= cfg['patience_evaluations']:
            break
        began = time.time()
        phase = model.configure_phase(epoch)
        model.set_training_mode()
        order = np.random.permutation(runtime.train_indices)
        task_losses, task_mses, rnc_losses, norms = [], [], [], []
        total_steps = (len(order)+cfg['batch_size']-1)//cfg['batch_size']
        for step, start in enumerate(range(0, len(order), cfg['batch_size'])):
            rows = order[start:start+cfg['batch_size']]
            batch = runtime.move(runtime.collate_indices(rows))
            target = batch['label'].flatten()
            optimizer.zero_grad(set_to_none=True)
            base, delta = model(batch)
            mse = (base + delta - target).square().mean()
            task_loss = mse + cfg['residual_l2'] * delta.square().mean()
            assert torch.isfinite(task_loss)
            task_loss.backward()
            weighted_value = 0.0
            if (step+1) % cfg['rnc']['interval'] == 0:
                positions = sampler.sample()
                rnc_rows = runtime.train_indices[positions]
                embeddings = []
                for rstart in range(0, len(rnc_rows), cfg['rnc']['microbatch_size']):
                    rbatch = runtime.move(runtime.collate_indices(
                        rnc_rows[rstart:rstart+cfg['rnc']['microbatch_size']]))
                    _, _, embedding = model(rbatch, return_contrast=True)
                    embeddings.append(embedding)
                affinity = torch.as_tensor(labels[positions], dtype=torch.float32, device='cuda')
                rnc_loss, _ = rank_n_contrast_loss(
                    torch.cat(embeddings), affinity, temperature=cfg['rnc']['temperature'], mode='standard')
                warmup = min(1.0, ((epoch-1)+(step+1)/total_steps)/cfg['rnc']['warmup_epochs'])
                weighted = cfg['rnc_weight'] * cfg['rnc']['interval'] * warmup * rnc_loss
                assert torch.isfinite(weighted)
                weighted.backward()
                weighted_value = float(weighted.detach())
                rnc_losses.append(float(rnc_loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            task_losses.append(float(task_loss.detach()) + weighted_value)
            task_mses.append(float(mse.detach())); norms.append(float(norm))
            if step % 200 == 0:
                extra = f' rnc={rnc_losses[-1]:.6f}' if rnc_losses else ''
                print(f'TRAIN variant={variant} fold={fold} epoch={epoch} phase={phase} '
                      f'step={step}/{total_steps} mse={task_mses[-1]:.6f}{extra}', flush=True)
        evaluated = epoch % cfg['evaluation_interval'] == 0
        choices = None
        improved = False
        if evaluated:
            train_base, train_delta = infer_rows(model, runtime, runtime.train_indices, cfg['eval_batch_size'])
            val_base, val_delta = infer_rows(model, runtime, runtime.val_indices, cfg['eval_batch_size'])
            chosen, choices = select_dynamic(runtime, train_base, train_delta, val_base, val_delta,
                                              epoch, cfg['lambdas'])
            improved = chosen['mse'] < best['mse'] - cfg['min_delta']
            if improved:
                best = chosen; stale = 0
                save(dest/'best.pt', {'model': model.state_dict(), 'selection': best, 'protocol': cfg,
                                      'variant': variant, 'seed': seed, 'fold': fold})
                np.savez_compressed(dest/'best_development_predictions.npz',
                                    train_rows=runtime.train_indices, train_base=train_base,
                                    train_delta=train_delta, val_rows=runtime.val_indices,
                                    val_base=val_base, val_delta=val_delta)
            elif epoch >= cfg['early_stop_start_epoch']:
                stale += 1
        record = {'epoch': epoch, 'phase': phase, **model.trainable_audit(),
                  'train_loss': float(np.mean(task_losses)), 'train_mse': float(np.mean(task_mses)),
                  'rnc_loss': float(np.mean(rnc_losses)), 'rnc_updates': len(rnc_losses),
                  'max_gradient_norm': max(norms), 'evaluated': evaluated,
                  'validation_candidates': choices, 'best': best.copy(), 'improved': improved,
                  'stale': stale, 'seconds': time.time()-began}
        history.append(record)
        save(dest/'latest.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'epoch': epoch, 'best': best, 'history': history, 'stale': stale,
                                'rng': rng_state(), 'protocol': cfg, 'variant': variant,
                                'seed': seed, 'fold': fold})
        dump(dest/'history.json', history)
        dump(dest/'progress.json', {'state': 'training', 'variant': variant, 'fold': fold,
                                    'seed': seed, **record, 'updated': time.time()})
        print('EPOCH', json.dumps(record), flush=True)
    dump(dest/'selection_locked.json', {'selection': best, 'validation_only': True,
         'finished': time.time(), 'training_seconds': time.time()-started})
    evaluate_run(dest, runtime, model, cfg, adapter_path)


def evaluate_run(dest, runtime, model, cfg, adapter_path):
    selected = torch.load(dest/'best.pt', map_location='cpu', weights_only=False)
    selection = selected['selection']
    cache = runtime.cache
    test_rows = runtime.test_indices
    test_drugs = cache['split_drugs']['test']
    labels = runtime.dataset.df.iloc[test_rows].label.to_numpy(dtype=np.float64)
    base_r1_matrix = R1(cache).correct(cache['base'], test_drugs)[0]
    base_values = query_values(base_r1_matrix, runtime, test_drugs, test_rows)
    if selection['source'] == 'locked_rank16':
        prediction_values = base_values.copy()
    elif selection['source'] == 'locked_pair_rnc':
        parent = pd.read_csv(PARENT_OUT/f'runs/rnc_standard_a001/seed_42/fold_{runtime.fold}/test_predictions.csv')
        lookup = dict(zip(parent.pair_index.astype(int), parent.rank16_pcim_R1.astype(float)))
        prediction_values = np.asarray([lookup[int(row)] for row in test_rows], dtype=np.float64)
    else:
        model.load_state_dict(selected['model']); model.eval()
        train_base, train_delta = infer_rows(model, runtime, runtime.train_indices, cfg['eval_batch_size'])
        test_base, test_delta = infer_rows(model, runtime, test_rows, cfg['eval_batch_size'])
        prediction = runtime.fill_matrix(cache['base'], runtime.train_indices,
                                         train_base + selection['lambda']*train_delta)
        prediction = runtime.fill_matrix(prediction, test_rows,
                                         test_base + selection['lambda']*test_delta)
        corrected = R1(cache).correct(prediction, test_drugs)[0]
        prediction_values = query_values(corrected, runtime, test_drugs, test_rows)
    table = pd.DataFrame({'pair_index': test_rows,
                          'drug_id': runtime.dataset.df.iloc[test_rows].drug_id.astype(str).to_numpy(),
                          'protein_id': runtime.dataset.df.iloc[test_rows].protein_id.astype(str).to_numpy(),
                          'label': labels, 'rank16_R1': base_values,
                          'rank16_pcim_rnc_unfreeze_R1': prediction_values})
    table.to_csv(dest/'test_predictions.csv', index=False)
    tests = {'rank16_R1': metric(base_values, labels),
             'new_R1': metric(prediction_values, labels)}
    high = labels >= cfg['rnc']['high_threshold']
    high_tests = {'rank16_R1': metric(base_values[high], labels[high]),
                  'new_R1': metric(prediction_values[high], labels[high])}
    by_drug = []
    for drug, group in table.groupby('drug_id'):
        b = float(np.mean((group.rank16_R1-group.label)**2))
        n = float(np.mean((group.rank16_pcim_rnc_unfreeze_R1-group.label)**2))
        by_drug.append({'drug_id': drug, 'base_mse': b, 'new_mse': n, 'new_minus_base': n-b})
    result = {'fold': runtime.fold, 'variant': selected['variant'], 'seed': selected['seed'],
              'selection': selection, 'test': tests, 'high_affinity_test': high_tests,
              'high_affinity_count': int(high.sum()), 'per_drug': by_drug,
              'adapter_initialization': str(adapter_path), 'adapter_sha256': sha(adapter_path),
              'backbone_checkpoint': cache['checkpoint'],
              'backbone_sha256': cache['checkpoint_sha256'],
              'label_poisoning_policy': 'test labels accessed only after selection_locked.json',
              'finished': time.time()}
    dump(dest/'result.json', result)
    print('TEST_RESULT', json.dumps(result), flush=True)


def aggregate():
    cfg = load_protocol()
    lock = (OUT/'aggregate.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        entries = [json.loads(p.read_text()) for p in sorted((OUT/'runs').glob('*/seed_*/fold_*/result.json'))]
        groups = {}
        for variant in cfg['variants']:
            selected = [x for x in entries if x['variant'] == variant]
            item = {'folds_completed': [x['fold'] for x in selected],
                    'enabled_folds': [x['fold'] for x in selected if x['selection']['source']=='finetuned'],
                    'selections': {str(x['fold']): x['selection'] for x in selected},
                    'complete': len(selected) == 5}
            if len(selected) == 5:
                item['macro'] = {name: {key: float(np.mean([r['test'][name][key] for r in selected]))
                                                for key in ['mse','rmse','mae','ci','rm2']}
                                 for name in ['rank16_R1','new_R1']}
                item['high_affinity_macro'] = {name: {key: float(np.mean([r['high_affinity_test'][name][key]
                                                                          for r in selected]))
                                                              for key in ['mse','rmse','mae','ci']}
                                               for name in ['rank16_R1','new_R1']}
            groups[variant] = item
        report = {'completed_runs': len(entries), 'planned_runs': 10,
                  'complete': len(entries) == 10, 'groups': groups,
                  'historical_controls': json.loads((OUT/'historical_controls.json').read_text()),
                  'updated': time.time()}
        dump(OUT/'summary.json', report)
        lines = ['# Progressive Rank16 unfreezing with standard pair-RNC', '',
                 f'Completed {len(entries)}/10 runs.', '',
                 '| Variant | Fold | Source | Lambda | Rank16+R1 MSE | New+R1 MSE |',
                 '|---|---:|---|---:|---:|---:|']
        for r in entries:
            lines.append(f'| {r["variant"]} | {r["fold"]} | {r["selection"]["source"]} | '
                         f'{r["selection"]["lambda"]} | {r["test"]["rank16_R1"]["mse"]:.6f} | '
                         f'{r["test"]["new_R1"]["mse"]:.6f} |')
        temp = OUT/f'REPORT.md.{os.getpid()}.tmp'
        temp.write_text('\n'.join(lines)+'\n'); os.replace(temp, OUT/'REPORT.md')
        return report
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


def worker(variant):
    cfg = load_protocol(); assert variant in cfg['variants']
    lock = (OUT/f'worker_{variant}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path = OUT/f'status_{variant}.json'
    try:
        for index, fold in enumerate(range(1, 6), 1):
            if (OUT/f'runs/{variant}/seed_42/fold_{fold}/result.json').exists():
                continue
            logpath = OUT/f'logs/{variant}_fold{fold}.log'
            with logpath.open('a') as log:
                process = subprocess.Popen([sys.executable, '-B', '-u', str(HERE/'train_experiment.py'),
                                            'train', '--variant', variant, '--fold', str(fold), '--seed', '42'],
                                           cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                           env=os.environ.copy())
                dump(status_path, {'state': 'running', 'variant': variant, 'fold': fold,
                                   'task_index': index, 'total_tasks': 5, 'pid': os.getpid(),
                                   'child_pid': process.pid, 'log': str(logpath), 'updated': time.time()})
                code = process.wait()
            if code:
                raise RuntimeError(f'{variant} fold{fold} exited {code}; see {logpath}')
            aggregate()
        aggregate()
        dump(status_path, {'state': 'completed', 'variant': variant, 'pid': os.getpid(),
                           'finished': time.time()})
    except Exception as exc:
        dump(status_path, {'state': 'failed', 'variant': variant, 'error': str(exc),
                           'traceback': traceback.format_exc(), 'pid': os.getpid(),
                           'updated': time.time()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train','worker','aggregate'])
    parser.add_argument('--variant', choices=['unfreeze_last','unfreeze_full'])
    parser.add_argument('--fold', type=int, choices=range(1,6))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == 'train':
        train_run(args.fold, args.variant, args.seed)
    elif args.action == 'worker':
        worker(args.variant)
    else:
        print(json.dumps(aggregate(), indent=2))
