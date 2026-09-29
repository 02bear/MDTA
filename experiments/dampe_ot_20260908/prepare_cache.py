"""Freeze the existing fold encoder; cache train/val entities, never test drugs."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import time
import torch
import pandas as pd
from common import PROJECT, OUTPUT, BASE, SPLITS, SOURCE_FILES, resolve, sha256, source_hashes, dump, validate_split
from datasets.davis_dataset_p13d import DavisDatasetP13D
from datasets.collate_p13d import mdta_collate_fn_p13d, move_batch_to_device
from train_p13d_earlystop import build_model


@torch.no_grad()
def prepare(fold, device):
    torch.set_num_threads(1)
    out = OUTPUT/f'fold_{fold}'/'cache'
    out.mkdir(parents=True, exist_ok=False)
    ckpath = BASE/f'fold_{fold}'/'best_model.pt'
    splitpath = SPLITS/f'fold_{fold}'/'split.json'
    ckhash = sha256(ckpath)
    ck = torch.load(ckpath, map_location='cpu', weights_only=False)
    if sha256(ckpath) != ckhash:
        raise RuntimeError('Checkpoint changed while being loaded')
    args = SimpleNamespace(**ck['args'])
    if resolve(args.split_json).resolve() != splitpath.resolve():
        raise ValueError('Baseline checkpoint belongs to another split')
    split = json.loads(splitpath.read_text())
    # Verify saved baseline training indices, not only the checkpoint path string.
    trained_split = json.loads((ckpath.parent/'split_indices.json').read_text())
    for part in ['train', 'val']:
        if trained_split[part+'_indices'] != split[part+'_indices']:
            raise ValueError('Checkpoint training split mismatch')
    ds = DavisDatasetP13D(pairs_csv=resolve(args.pairs_csv),
        drug_1d_dir=resolve(args.drug_1d_dir), protein_1d_dir=resolve(args.protein_1d_dir),
        protein_3d_dir=resolve(args.protein_3d_dir), drug_3d_dir=resolve(args.drug_3d_dir),
        use_drug_3d=True)
    raw = pd.read_csv(resolve(args.pairs_csv))
    if len(raw) != len(ds.df) or not (raw.drug_id.astype(str).tolist() == ds.df.drug_id.tolist()
            and raw.protein_id.astype(str).tolist() == ds.df.protein_id.tolist()):
        raise ValueError('Filtering changed row identity; refuse to reinterpret split indices')
    split_audit = validate_split(ds.df, split)
    model = build_model(args, torch.device(device))
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model.requires_grad_(False)
    model.eval()
    train = ds.df.iloc[split['train_indices']]
    used = ds.df.iloc[split['train_indices']+split['val_indices']]
    entity_data = {}
    files_hash = {}
    started = time.monotonic()
    for kind in ['drug', 'protein']:
        col = kind+'_id'
        ids = sorted(used[col].unique())
        representatives = used.reset_index().groupby(col)['index'].first().to_dict()
        f1, f3 = [], []
        for j, eid in enumerate(ids):
            sample = ds[int(representatives[eid])]
            batch = move_batch_to_device(mdta_collate_fn_p13d([sample]), torch.device(device))
            h1 = getattr(model, kind+'_1d_encoder')(batch[kind+'_1d'])
            h3 = getattr(model, kind+'_3d_encoder')(batch[kind+'_3d'])
            if not torch.isfinite(h1).all() or not torch.isfinite(h3).all():
                raise ValueError('Nonfinite '+kind+' '+eid)
            f1.append(h1.cpu()[0]); f3.append(h3.cpu()[0])
            for dim in ['1d', '3d']:
                path = resolve(getattr(args, kind+'_'+dim+'_dir'))/(eid+'.pt')
                files_hash[str(path.relative_to(PROJECT))] = sha256(path)
            if j % 50 == 0:
                print(f'CACHE {kind} {j+1}/{len(ids)} elapsed={time.monotonic()-started:.1f}s', flush=True)
        entity_data[kind] = dict(ids=ids, train_ids=sorted(train[col].unique()),
                                 h1=torch.stack(f1), h3=torch.stack(f3))
    # Verify original whole-model forward equals the cached path with original heads.
    verification = []
    for idx in split['train_indices'][:2]+split['val_indices'][:2]:
        sample = ds[idx]
        batch = move_batch_to_device(mdta_collate_fn_p13d([sample]), torch.device(device))
        actual = model(batch)
        features = []
        for kind in ['drug', 'protein']:
            e = entity_data[kind]; k = e['ids'].index(sample[kind+'_id'])
            z = getattr(model, kind+'_fusion')([e['h1'][k:k+1].to(device), e['h3'][k:k+1].to(device)])
            features.append(z)
        cached = model.decoder(torch.cat(features, -1))
        error = float((actual-cached).abs().max())
        torch.testing.assert_close(actual, cached, atol=1e-5, rtol=1e-5)
        verification.append(error)
    hashes = source_hashes()
    for f in SOURCE_FILES:
        target = out/'source_snapshot'/f
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/f, target)
        if sha256(target) != hashes[f]:
            raise RuntimeError('Source changed during snapshot')
    # Persist exact frozen encoder state, making the feature cache reproducible.
    encoder_state = {k:v.detach().cpu() for k,v in model.state_dict().items() if '_encoder.' in k}
    torch.save(encoder_state, out/'frozen_encoder_state.pt')
    pairs = {}
    for part in ['train', 'val']:
        frame = ds.df.iloc[split[part+'_indices']]
        pairs[part] = dict(indices=split[part+'_indices'],
            drug_id=frame.drug_id.tolist(), protein_id=frame.protein_id.tolist(),
            label=torch.tensor(frame.label.tolist(), dtype=torch.float32).view(-1,1))
    cache = dict(fold=fold, entities=entity_data, pairs=pairs,
                 source_sha256=hashes, baseline_args=vars(args),
                 checkpoint_sha256=ckhash, split_sha256=sha256(splitpath))
    torch.save(cache, out/'features.pt')
    metadata = dict(fold=fold, baseline_checkpoint=str(ckpath), checkpoint_sha256=ckhash,
        checkpoint_epoch=ck['epoch'], encoder_checkpoint_selection='original validation early stopping',
        split_path=str(splitpath), split_sha256=sha256(splitpath),
        pairs_sha256=sha256(resolve(args.pairs_csv)), split_audit=split_audit,
        source_sha256=hashes, feature_file_sha256=files_hash,
        cache_sha256=sha256(out/'features.pt'),
        frozen_encoder_sha256=sha256(out/'frozen_encoder_state.pt'),
        encoder_eval_mode=True, encoder_trainable_parameters=0,
        cached_parts=['train','val'], test_drug_features_cached=False,
        original_vs_cache_max_abs_errors=verification,
        elapsed_seconds=time.monotonic()-started,
        shapes={k:{z:list(v[z].shape) for z in ['h1','h3']} for k,v in entity_data.items()})
    dump(out/'metadata.json', metadata)
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['feature_file_sha256','source_sha256']}, indent=2), flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--fold',type=int,default=1);p.add_argument('--device',default='cuda:0')
    a=p.parse_args(); prepare(a.fold,a.device)
