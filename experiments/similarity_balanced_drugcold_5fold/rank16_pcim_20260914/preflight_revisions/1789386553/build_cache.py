"""Extract frozen features once per entity and verify against historical predictions."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from common import HERE, PROJECT, OUT, BASE_OUT, SPLITS, sha, dump, save, seed_all, locked_modules, verify_manifest, R1

AA3 = ['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR']
AA = {name:i for i,name in enumerate(AA3)}
AA['MSE'] = AA['MET']


def build(fold):
    verify_manifest()
    destination = OUT/f'cache/fold_{fold}.pt'
    if destination.exists():
        audit = json.loads((OUT/f'cache/fold_{fold}_audit.json').read_text())
        assert sha(destination) == audit['cache_sha256']
        print('CACHE_REUSED', fold, flush=True)
        return
    h,t = locked_modules()
    reference = json.loads((HERE/f'references/fold_{fold}.json').read_text())
    ck = torch.load(reference['checkpoint'], map_location='cpu', weights_only=False)
    assert sha(reference['checkpoint']) == reference['checkpoint_sha256']
    args = SimpleNamespace(**ck['args'])
    assert Path(args.split_json).resolve() == (SPLITS/f'fold_{fold}/split.json').resolve()
    split = json.loads(Path(args.split_json).read_text())
    seed_all(42)
    model = h.build_model(args, torch.device('cuda'))
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    dataset, _, _, _, _ = t.build_dataloaders(args)
    df = dataset.df.copy()
    raw = pd.read_csv(PROJECT/args.pairs_csv, dtype={'drug_id':str,'protein_id':str})
    assert len(df) == 30056
    assert df[['drug_id','protein_id']].equals(raw[['drug_id','protein_id']])
    sim = np.load(HERE/'entity_similarities.npz', allow_pickle=True)
    drugs = list(map(str, sim['drug_ids'].tolist()))
    proteins = list(map(str, sim['protein_ids'].tolist()))
    di,pi = {x:i for i,x in enumerate(drugs)}, {x:i for i,x in enumerate(proteins)}
    rows = np.full((len(drugs),len(proteins)), -1, dtype=np.int64)
    for i,r in enumerate(df.itertuples(index=False)): rows[di[r.drug_id],pi[r.protein_id]] = i
    assert (rows>=0).all() and np.unique(rows).size == len(df)
    # Historical dataset emits float32 pKd labels before conversion to float64.
    labels = df.label.to_numpy(dtype=np.float32).astype(np.float64)[rows]
    input_hashes, atom_cache, residue_cache, aa_cache, dglobal, pglobal = {}, [], [], [], [], []
    local_api_errors, unknown = [], 0
    # Run each molecule independently; no graph connects different entities.
    with torch.inference_mode():
        for d in drugs:
            gpath = PROJECT/args.drug_3d_dir/f'{d}.pt'
            xpath = PROJECT/args.drug_1d_dir/f'{d}.pt'
            for path in [gpath,xpath]: input_hashes[str(path)] = sha(path)
            obj = torch.load(gpath, map_location='cpu', weights_only=False)
            x1 = torch.load(xpath, map_location='cpu', weights_only=False)['mean'].float()[None].cuda()
            graph = {k:obj[k].cuda() for k in ['x','pos','edge_index']}
            graph['x'], graph['pos'] = graph['x'].float(), graph['pos'].float()
            graph['batch'] = torch.zeros(len(obj['x']), dtype=torch.long, device='cuda')
            out = model.drug_3d_encoder(graph, return_node=True)
            normal = model.drug_3d_encoder(graph)
            local_api_errors.append(float((out['graph_feat']-normal).abs().max()))
            atom_cache.append(out['node_feat'].cpu().clone())
            dglobal.append(model.drug_fusion([model.drug_1d_encoder(x1), out['graph_feat']]).cpu().clone()[0])
        for index,p in enumerate(proteins):
            gpath = PROJECT/args.protein_3d_dir/f'{p}.pt'
            xpath = PROJECT/args.protein_1d_dir/f'{p}.pt'
            for path in [gpath,xpath]: input_hashes[str(path)] = sha(path)
            obj = torch.load(gpath, map_location='cpu', weights_only=False)
            assert len(obj['residue_meta']) == len(obj['node_s']), p
            types = torch.tensor([AA.get(str(m['resname']).upper(),20) for m in obj['residue_meta']], dtype=torch.long)
            unknown += int((types==20).sum())
            aa_cache.append(types)
            x1 = torch.load(xpath, map_location='cpu', weights_only=False)['mean'].float()[None].cuda()
            graph = {k:obj[k].cuda() for k in ['node_s','node_v','coords','edge_index','edge_s','edge_v']}
            graph['batch'] = torch.zeros(len(types), dtype=torch.long, device='cuda')
            out = model.protein_3d_encoder(graph, return_node=True)
            # Several lengths are checked, including the first and last protein.
            if index in (0,100,200,300,441):
                normal = model.protein_3d_encoder(graph)
                local_api_errors.append(float((out['graph_feat']-normal).abs().max()))
            residue_cache.append(out['node_feat'].cpu().clone())
            pglobal.append(model.protein_fusion([model.protein_1d_encoder(x1),out['graph_feat']]).cpu().clone()[0])
            if index % 50 == 0: print('CACHE_PROTEIN', fold,index,len(proteins),flush=True)
        dg,pg = torch.stack(dglobal),torch.stack(pglobal)
        reconstruction = np.empty(len(df), dtype=np.float64)
        for start in range(0,len(df),16):
            part = df.iloc[start:start+16]
            d = dg[[di[x] for x in part.drug_id]].cuda()
            p = pg[[pi[x] for x in part.protein_id]].cuda()
            reconstruction[start:start+len(part)] = model.decoder(torch.cat([d,p],-1)).flatten().cpu().numpy()
    # Use historical f0 predictions as the protected skip, avoiding batch-roundoff drift.
    saved = np.full(len(df), np.nan, dtype=np.float64)
    hist = BASE_OUT/'bilinear_rank16_complete_20260914'/f'evaluation/fold_{fold}'
    train = np.load(hist/'train_predictions.npz', allow_pickle=True)
    saved[train['pair_index']] = train['prediction']
    val = np.load(Path(reference['checkpoint']).parent/'best_val_predictions.npz', allow_pickle=True)
    saved[val['indices']] = val['y_pred']
    test = pd.read_csv(hist/'test_predictions.csv', dtype={'drug_id':str,'protein_id':str})
    saved[test.pair_index] = test.rank16
    assert np.isfinite(saved).all()
    error = float(np.abs(reconstruction-saved).max())
    assert error < 1e-4, f'Frozen feature reconstruction differs: {error}'
    assert max(local_api_errors) == 0.0
    split_drugs = {part:[di[str(x)] for x in split[part+'_drugs']] for part in ['train','val','test']}
    for part in split_drugs:
        assert sorted(rows[split_drugs[part]].ravel().tolist()) == sorted(split[part+'_indices'])
    assert set(split_drugs['train']).isdisjoint(split_drugs['val'])
    assert set(split_drugs['train']).isdisjoint(split_drugs['test'])
    assert set(split_drugs['val']).isdisjoint(split_drugs['test'])
    data = dict(fold=fold, drugs=drugs, proteins=proteins, pair_indices=rows, labels=labels,
                base=saved[rows], atoms=atom_cache, residues=residue_cache, aa=aa_cache,
                drug_global=dg, protein_global=pg, split_drugs=split_drugs,
                similarity=sim['drug_similarity'], r1_config=reference['r1_config'],
                checkpoint=reference['checkpoint'], checkpoint_sha256=reference['checkpoint_sha256'])
    r1 = R1(data)
    query = split_drugs['test']
    replay,_,_ = r1.correct(data['base'],query)
    testlookup = dict(zip(test.pair_index, test.rank16_R1))
    expected = np.array([testlookup[i] for i in rows[query].ravel()]).reshape(replay.shape)
    r1_error = float(np.abs(replay-expected).max())
    assert r1_error < 1e-10, r1_error
    poisoned = dict(data)
    poisoned['labels'] = labels.copy()
    poisoned['labels'][split_drugs['val']+split_drugs['test']] = np.nan
    assert np.array_equal(R1(poisoned).correct(data['base'],query)[0],replay)
    assert all(torch.isfinite(x).all() for x in atom_cache+residue_cache+[dg,pg])
    save(destination,data)
    dump(OUT/f'cache/fold_{fold}_audit.json', dict(fold=fold, passed=True,
         cache_sha256=sha(destination), checkpoint_sha256=reference['checkpoint_sha256'],
         split_sha256=sha(args.split_json), frozen_prediction_max_error=error,
         return_node_global_error=max(local_api_errors), historical_r1_max_error=r1_error,
         nontraining_label_poisoning_passed=True, unknown_residue_count=unknown,
         max_atoms=max(map(len,atom_cache)),max_residues=max(map(len,residue_cache)),
         source_inputs_sha256=input_hashes, baseline_predictions='historical skip values; reconstructed from frozen features within 1e-4'))
    print('CACHE_PASSED',fold,error,r1_error,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold',type=int,required=True,choices=range(1,6))
    build(parser.parse_args().fold)
