"""Exactly the original ConcatFusion and Decoder on cached frozen features."""
import torch
from torch import nn
from common import PROJECT
from models.fusion import ConcatFusion
from models.decoder import Decoder
from ot_alignment import OTAlignment


class FrozenOTHead(nn.Module):
    def __init__(self, drug_mapping, protein_mapping, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.drug_ot = OTAlignment(drug_mapping)
        self.protein_ot = OTAlignment(protein_mapping)
        self.drug_fusion = ConcatFusion([hidden_dim, hidden_dim], out_dim=hidden_dim,
                                       hidden_dim=hidden_dim*2, dropout=dropout)
        self.protein_fusion = ConcatFusion([hidden_dim, hidden_dim], out_dim=hidden_dim,
                                          hidden_dim=hidden_dim*2, dropout=dropout)
        self.decoder = Decoder(input_dim=hidden_dim*2, hidden_dim=hidden_dim,
                               dropout=dropout, task='regression')

    def forward(self, batch):
        drug = self.drug_fusion([batch['drug_1d'], self.drug_ot(batch['drug_3d'])])
        protein = self.protein_fusion([batch['protein_1d'], self.protein_ot(batch['protein_3d'])])
        return self.decoder(torch.cat([drug, protein], dim=-1))
