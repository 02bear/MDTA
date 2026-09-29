"""Original P13D concat model with detached, periodically replaced OT maps."""
import torch
import torch.nn as nn

from models.drug_1d_encoder import Drug1DEncoder
from models.drug_3d_egnn_encoder import Drug3DEGNNEncoder
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_egnn_encoder import Protein3DEGNNEncoder
from models.fusion import ConcatFusion
from models.decoder import Decoder


class MyModelMDTAP13DPeriodicOT(nn.Module):
    def __init__(self, drug_1d_in_dim=768, drug_3d_node_in_dim=10,
                 protein_1d_in_dim=1280, protein_3d_node_s_dim=6,
                 protein_3d_node_v_dim=3, hidden_dim=128, dropout=0.1,
                 task="regression"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drug_1d_encoder = Drug1DEncoder(drug_1d_in_dim, hidden_dim)
        self.drug_3d_encoder = Drug3DEGNNEncoder(
            node_in_dim=drug_3d_node_in_dim, hidden_dim=hidden_dim,
            out_dim=hidden_dim, n_layers=3, dropout=dropout)
        self.drug_fusion = ConcatFusion(
            [hidden_dim, hidden_dim], out_dim=hidden_dim,
            hidden_dim=hidden_dim * 2, dropout=dropout)
        self.protein_1d_encoder = Protein1DEncoder(protein_1d_in_dim, hidden_dim)
        self.protein_3d_encoder = Protein3DEGNNEncoder(
            node_s_dim=protein_3d_node_s_dim, hidden_dim=hidden_dim,
            out_dim=hidden_dim, dropout=dropout, n_layers=3)
        self.protein_fusion = ConcatFusion(
            [hidden_dim, hidden_dim], out_dim=hidden_dim,
            hidden_dim=hidden_dim * 2, dropout=dropout)
        self.decoder = Decoder(hidden_dim * 2, hidden_dim, dropout, task)
        self.register_buffer("drug_ot_map", torch.eye(hidden_dim))
        self.register_buffer("protein_ot_map", torch.eye(hidden_dim))
        self.register_buffer("ot_enabled", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def set_ot_maps(self, drug_map, protein_map, enabled=True):
        for name, value in [("drug_ot_map", drug_map),
                            ("protein_ot_map", protein_map)]:
            value = torch.as_tensor(value, device=getattr(self, name).device,
                                    dtype=getattr(self, name).dtype)
            if value.shape != (self.hidden_dim, self.hidden_dim):
                raise ValueError(f"{name} has shape {tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains nonfinite values")
            getattr(self, name).copy_(value)
        self.ot_enabled.fill_(enabled)

    def encode_modalities(self, batch):
        return {
            "drug_1d": self.drug_1d_encoder(batch["drug_1d"]),
            "drug_3d": self.drug_3d_encoder(batch["drug_3d"]),
            "protein_1d": self.protein_1d_encoder(batch["protein_1d"]),
            "protein_3d": self.protein_3d_encoder(batch["protein_3d"]),
        }

    def forward(self, batch):
        h = self.encode_modalities(batch)
        if bool(self.ot_enabled.item()):
            h["drug_3d"] = h["drug_3d"] @ self.drug_ot_map
            h["protein_3d"] = h["protein_3d"] @ self.protein_ot_map
        drug = self.drug_fusion([h["drug_1d"], h["drug_3d"]])
        protein = self.protein_fusion([h["protein_1d"], h["protein_3d"]])
        return self.decoder(torch.cat([drug, protein], dim=-1))
