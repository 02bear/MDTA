"""Raw-input Rank16 forward joined to the existing pair interaction adapter."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset

from common import CACHE_SOURCE, PARENT_OUT, load_protocol, locked_modules, sha
from pair_interaction import PairResidual


class FoldRuntime:
    """Fold-locked raw dataset plus mappings to the cached drug x protein grid."""
    def __init__(self, fold, device='cuda'):
        self.fold = int(fold)
        self.device = torch.device(device)
        self.cache = torch.load(CACHE_SOURCE/f'fold_{fold}.pt', map_location='cpu', weights_only=False)
        checkpoint = Path(self.cache['checkpoint'])
        assert sha(checkpoint) == self.cache['checkpoint_sha256']
        self.checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.args = SimpleNamespace(**self.checkpoint['args'])
        self.impl, self.base = locked_modules()
        self.dataset, train_set, val_set, _, _ = self.base.build_dataloaders(self.args)
        split = json.loads(Path(self.args.split_json).read_text())
        assert list(train_set.indices) == split['train_indices']
        assert list(val_set.indices) == split['val_indices']
        self.train_indices = np.asarray(split['train_indices'], dtype=np.int64)
        self.val_indices = np.asarray(split['val_indices'], dtype=np.int64)
        self.test_indices = np.asarray(split['test_indices'], dtype=np.int64)
        assert len(self.dataset) == self.cache['pair_indices'].size
        self.drug_to_index = {str(x): i for i, x in enumerate(self.cache['drugs'])}
        self.protein_to_index = {str(x): i for i, x in enumerate(self.cache['proteins'])}
        self.row_to_drug = np.empty(len(self.dataset), dtype=np.int64)
        self.row_to_protein = np.empty(len(self.dataset), dtype=np.int64)
        for d in range(len(self.cache['drugs'])):
            for p in range(len(self.cache['proteins'])):
                row = int(self.cache['pair_indices'][d, p])
                self.row_to_drug[row] = d
                self.row_to_protein[row] = p
        frame = self.dataset.df
        for row in [0, len(frame)//2, len(frame)-1]:
            assert self.drug_to_index[str(frame.iloc[row].drug_id)] == self.row_to_drug[row]
            assert self.protein_to_index[str(frame.iloc[row].protein_id)] == self.row_to_protein[row]
            d, p = self.row_to_drug[row], self.row_to_protein[row]
            assert abs(float(frame.iloc[row].label)-float(self.cache['labels'][d, p])) < 1e-6

    def loader(self, indices, batch_size, shuffle=False):
        return DataLoader(Subset(self.dataset, list(map(int, indices))), batch_size=batch_size,
                          shuffle=shuffle, num_workers=0, pin_memory=True,
                          collate_fn=self.base.mdta_collate_fn_p13d)

    def collate_indices(self, indices):
        return self.base.mdta_collate_fn_p13d([self.dataset[int(i)] for i in indices])

    def move(self, batch):
        moved = self.base.move_batch_to_device(batch, self.device)
        aa = [self.cache['aa'][self.protein_to_index[str(pid)]].long() for pid in batch['protein_id']]
        moved['aa'] = pad_sequence(aa, batch_first=True, padding_value=20).to(self.device)
        return moved

    def fill_matrix(self, matrix, row_indices, values):
        result = np.array(matrix, copy=True)
        row_indices = np.asarray(row_indices, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        assert len(row_indices) == len(values)
        result[self.row_to_drug[row_indices], self.row_to_protein[row_indices]] = values
        return result


class EndToEndPairRNC(nn.Module):
    """Trainable Rank16 backbone feeding the current PCIM/RNC adapter."""
    def __init__(self, runtime, variant, adapter_checkpoint):
        super().__init__()
        assert variant in ('unfreeze_last', 'unfreeze_full')
        cfg = load_protocol()
        self.variant = variant
        self.backbone = runtime.impl.build_model(runtime.args, runtime.device)
        self.backbone.load_state_dict(runtime.checkpoint['model_state_dict'], strict=True)
        self.interaction = PairResidual(
            'bidirectional', top_k=cfg['top_k'], max_delta=cfg['max_delta'],
            selection='coverage', coverage_budget=cfg['coverage_budget'],
            cross_max_scale=cfg['cross_attention']['max_residual_scale'],
            enable_contrast=True, contrast_dim=cfg['rnc']['projection_dim']).to(runtime.device)
        adapter = torch.load(adapter_checkpoint, map_location='cpu', weights_only=False)
        self.interaction.load_state_dict(adapter['model'], strict=True)
        self.adapter_selection = adapter['selection']
        self.current_phase = None
        self.configure_phase(1)

    @staticmethod
    def _pad_nodes(features, batch_index):
        count = int(batch_index.max().item()) + 1
        sequence = [features[batch_index == i] for i in range(count)]
        lengths = torch.tensor([len(x) for x in sequence], device=features.device)
        padded = pad_sequence(sequence, batch_first=True)
        mask = torch.arange(padded.size(1), device=features.device)[None] < lengths[:, None]
        return padded, mask

    def _stage_modules(self):
        b = self.backbone
        head = [b.decoder, b.drug_fusion, b.protein_fusion,
                b.drug_3d_encoder.out_proj, b.protein_3d_encoder.out_proj]
        last = [b.drug_3d_encoder.layers[2], b.protein_3d_encoder.layers[2]]
        one_d = [b.drug_1d_encoder, b.protein_1d_encoder]
        early = [b.drug_3d_encoder.input_proj, b.protein_3d_encoder.input_proj,
                 b.drug_3d_encoder.layers[0], b.drug_3d_encoder.layers[1],
                 b.protein_3d_encoder.layers[0], b.protein_3d_encoder.layers[1]]
        return head, last, one_d, early

    def configure_phase(self, epoch):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        head, last, one_d, early = self._stage_modules()
        if epoch >= 3:
            for module in head + last + one_d:
                module.requires_grad_(True)
        if self.variant == 'unfreeze_full' and epoch >= 6:
            for module in early:
                module.requires_grad_(True)
        self.interaction.requires_grad_(True)
        self.current_phase = 'adapter' if epoch < 3 else ('full' if self.variant == 'unfreeze_full' and epoch >= 6 else 'last')
        return self.current_phase

    def set_training_mode(self):
        self.interaction.train()
        self.backbone.eval()
        if self.current_phase == 'full':
            self.backbone.train()
        elif self.current_phase == 'last':
            head, last, one_d, _ = self._stage_modules()
            for module in head + last + one_d:
                module.train()

    def optimizer_groups(self, cfg):
        head, last, one_d, early = self._stage_modules()
        groups = []
        seen = set()
        def add(name, modules, lr):
            params = []
            for module in modules:
                for parameter in module.parameters():
                    if id(parameter) not in seen:
                        seen.add(id(parameter)); params.append(parameter)
            assert params
            groups.append({'params': params, 'lr': lr, 'name': name})
        add('interaction', [self.interaction], cfg['learning_rates']['interaction'])
        add('rank_head_fusion_out', head, cfg['learning_rates']['rank_head_fusion_out'])
        add('egnn_last', last, cfg['learning_rates']['egnn_last'])
        add('one_d', one_d, cfg['learning_rates']['one_d'])
        add('egnn_early', early, cfg['learning_rates']['egnn_early'])
        expected = {id(p) for p in self.parameters()}
        assert seen == expected, (len(seen), len(expected))
        return groups

    def trainable_audit(self):
        return {'phase': self.current_phase,
                'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'total': sum(p.numel() for p in self.parameters()),
                'backbone_trainable': sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
                'interaction_trainable': sum(p.numel() for p in self.interaction.parameters() if p.requires_grad)}

    def forward(self, batch, return_contrast=False):
        drug_1d = self.backbone.drug_1d_encoder(batch['drug_1d'])
        drug_3d = self.backbone.drug_3d_encoder(batch['drug_3d'], return_node=True)
        drug_global = self.backbone.drug_fusion([drug_1d, drug_3d['graph_feat']])
        protein_1d = self.backbone.protein_1d_encoder(batch['protein_1d'])
        protein_3d = self.backbone.protein_3d_encoder(batch['protein_3d'], return_node=True)
        protein_global = self.backbone.protein_fusion([protein_1d, protein_3d['graph_feat']])
        base = self.backbone.decoder(torch.cat([drug_global, protein_global], -1)).squeeze(-1)
        atoms, atom_mask = self._pad_nodes(drug_3d['node_feat'], drug_3d['batch'])
        residues, residue_mask = self._pad_nodes(protein_3d['node_feat'], protein_3d['batch'])
        assert residues.size(1) == batch['aa'].size(1)
        interaction_batch = {'atoms': atoms, 'atom_mask': atom_mask,
                             'residues': residues, 'residue_mask': residue_mask,
                             'aa': batch['aa'], 'drug_global': drug_global,
                             'protein_global': protein_global}
        if return_contrast:
            delta, embedding = self.interaction(interaction_batch, return_contrast=True)
            return base, delta, embedding
        return base, self.interaction(interaction_batch)


def adapter_checkpoint_for_fold(fold):
    run = PARENT_OUT/f'runs/rnc_standard_a001/seed_42/fold_{fold}'
    result = json.loads((run/'result.json').read_text())
    path = run/('best_active.pt' if result['baseline_fallback'] else 'best.pt')
    assert path.exists()
    return path, result


def locked_pair_development(runtime):
    """Exact stage-1 chosen prediction on train/validation, for safe fallback."""
    path = PARENT_OUT/f'runs/rnc_standard_a001/seed_42/fold_{runtime.fold}'
    result = json.loads((path/'result.json').read_text())
    prediction = np.array(runtime.cache['base'], copy=True)
    if result['baseline_fallback']:
        return prediction, result
    saved = np.load(path/'best_development_predictions.npz')
    scale = float(saved['lambda_value'])
    for part in ('train', 'val'):
        drugs = saved[f'{part}_drugs'].astype(np.int64)
        prediction[drugs] += scale * saved[f'{part}_delta']
    return prediction, result
