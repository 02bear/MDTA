# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.drug_encoder import Drug1DEncoder
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_encoder import Protein3DEncoder
from models.fusion import ConcatFusion
from models.decoder import Decoder


class MyModelMDTAP13DCLIP(nn.Module):
    """
    完全按照DrugCLIP论文实现的标准对比学习框架
    只使用batch内的正负样本对，不使用亲和力矩阵
    """

    def __init__(
            self,
            drug_1d_in_dim=768,
            protein_1d_in_dim=1280,
            protein_3d_node_s_dim=6,
            protein_3d_node_v_dim=3,
            hidden_dim=128,
            contrastive_dim=128,
            dropout=0.1,
            task="regression",
            temperature_init=0.07,  # DrugCLIP使用的温度参数
    ):
        super().__init__()

        # -------- Encoders --------
        self.drug_1d_encoder = Drug1DEncoder(
            input_dim=drug_1d_in_dim,
            hidden_dim=hidden_dim
        )

        self.protein_1d_encoder = Protein1DEncoder(
            input_dim=protein_1d_in_dim,
            hidden_dim=hidden_dim
        )

        self.protein_3d_encoder = Protein3DEncoder(
            node_s_dim=protein_3d_node_s_dim,
            node_v_dim=protein_3d_node_v_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout
        )

        self.protein_fusion = ConcatFusion(
            input_dims=[hidden_dim, hidden_dim],
            out_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            dropout=dropout
        )

        # -------- Decoder --------
        self.decoder = Decoder(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task
        )

        # -------- CLIP projection heads --------
        self.drug_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, contrastive_dim),
        )

        self.protein_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, contrastive_dim),
        )

        # learnable temperature / logit scale (按照DrugCLIP论文)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature_init), dtype=torch.float32)
        )

    def compute_clip_loss(self, drug_feat, protein_feat):
        """
        按照DrugCLIP论文实现的标准对比学习损失
        包含两个方向：Drug-to-Protein 和 Protein-to-Drug
        """
        # 归一化特征
        z_d = F.normalize(self.drug_proj(drug_feat), dim=-1)  # [B, C]
        z_p = F.normalize(self.protein_proj(protein_feat), dim=-1)  # [B, C]

        # 计算相似度矩阵 [B, B]
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * torch.matmul(z_d, z_p.t())  # [B, B]

        # 标准对比学习标签 (对角线为正样本)
        labels = torch.arange(logits.size(0), device=logits.device)

        # Drug-to-Protein loss (每个drug应该匹配对应的protein)
        loss_d2p = F.cross_entropy(logits, labels)

        # Protein-to-Drug loss (每个protein应该匹配对应的drug)
        loss_p2d = F.cross_entropy(logits.t(), labels)

        # 总CLIP损失 (两个方向的平均)
        clip_loss = 0.5 * (loss_d2p + loss_p2d)

        # 计算平均hard negative分数 (用于监控)
        with torch.no_grad():
            # 移除对角线元素
            mask = torch.eye(logits.size(0), device=logits.device).bool()
            logits_masked = logits.clone()
            logits_masked[mask] = -1e9

            # 找出每个样本最难的负样本
            hard_neg_scores = torch.max(logits_masked, dim=1)[0]
            mean_hard_neg_score = torch.mean(hard_neg_scores)

        return clip_loss, logits, mean_hard_neg_score

    def forward(self, batch, return_aux=True):
        # -------- main features --------
        drug_feat = self.drug_1d_encoder(batch["drug_1d"])  # [B, H]

        protein_1d_feat = self.protein_1d_encoder(batch["protein_1d"])  # [B, H]
        protein_3d_feat = self.protein_3d_encoder(batch["protein_3d"])  # [B, H]
        protein_feat = self.protein_fusion([protein_1d_feat, protein_3d_feat])  # [B, H]

        # -------- regression prediction --------
        pair_feat = torch.cat([drug_feat, protein_feat], dim=-1)
        pred = self.decoder(pair_feat)

        # -------- CLIP loss --------
        clip_loss, clip_logits, mean_hard_neg_score = self.compute_clip_loss(
            drug_feat=drug_feat,
            protein_feat=protein_feat
        )

        if return_aux:
            aux = {
                "clip_loss": clip_loss,
                "clip_logits": clip_logits,
                "logit_scale": self.logit_scale.exp().detach(),
                "drug_feat": drug_feat,
                "protein_feat": protein_feat,
                "mean_valid_neg": mean_hard_neg_score.detach(),  # 重命名变量
            }
            return pred, aux
        else:
            return pred