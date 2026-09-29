# -*- coding: utf-8 -*-
"""
E5: Rich-attribute-aware atom-residue pair interaction.

保持E2全局分支、10维Drug EGNN、口袋选择和局部残差结构不变。

唯一变化：
将每个药物原子的43维rich_x直接加入atom-residue pair表示，
使pair_score_mlp和pair_feat_mlp直接利用丰富化学属性。
"""

import torch
import torch.nn as nn

from models.model_p13d_finegrained_residual import (
    AtomResidueInteraction,
    MyModelMDTAP13DFineGrained as E2Model,
)


class RichAttributeAwareAtomResidueInteraction(
    AtomResidueInteraction
):
    """
    E2 Pair输入:
        [atom, residue, atom*residue, |atom-residue|]
        维度 = 4H

    E5 Pair输入:
        [atom, residue, atom*residue, |atom-residue|, rich_atom]
        维度 = 4H + 43
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        rich_atom_dim: int = 43,
    ):
        # 先严格初始化原E2交互模块
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        if rich_atom_dim <= 0:
            raise ValueError(
                f"rich_atom_dim must be positive, "
                f"got {rich_atom_dim}"
            )

        self.rich_atom_dim = int(rich_atom_dim)
        self.base_pair_dim = hidden_dim * 4
        self.e5_pair_dim = (
            self.base_pair_dim + self.rich_atom_dim
        )

        # 保存RNG状态，避免扩展Linear影响后续随机序列
        cpu_rng_state = torch.get_rng_state()

        old_score_first = self.pair_score_mlp[0]
        old_feat_first = self.pair_feat_mlp[0]

        new_score_first = nn.Linear(
            self.e5_pair_dim,
            hidden_dim,
            bias=old_score_first.bias is not None,
        )

        new_feat_first = nn.Linear(
            self.e5_pair_dim,
            hidden_dim,
            bias=old_feat_first.bias is not None,
        )

        with torch.no_grad():
            # 原E2的4H权重完整保留
            new_score_first.weight.zero_()
            new_score_first.weight[
                :, :self.base_pair_dim
            ].copy_(old_score_first.weight)

            new_feat_first.weight.zero_()
            new_feat_first.weight[
                :, :self.base_pair_dim
            ].copy_(old_feat_first.weight)

            # 新增43维对应的权重保持为0
            # 因而E5初始化时严格等价于E2

            if old_score_first.bias is not None:
                new_score_first.bias.copy_(
                    old_score_first.bias
                )

            if old_feat_first.bias is not None:
                new_feat_first.bias.copy_(
                    old_feat_first.bias
                )

        self.pair_score_mlp[0] = new_score_first
        self.pair_feat_mlp[0] = new_feat_first

        torch.set_rng_state(cpu_rng_state)

    def copy_e2_parameters(
        self,
        source: AtomResidueInteraction,
    ):
        """
        将已经初始化好的E2交互模块参数复制到E5，
        新增43维权重列保持为0。
        """
        with torch.no_grad():
            self.atom_to_pocket_attn.load_state_dict(
                source.atom_to_pocket_attn.state_dict()
            )
            self.atom_norm.load_state_dict(
                source.atom_norm.state_dict()
            )
            self.out_norm.load_state_dict(
                source.out_norm.state_dict()
            )

            # Pair score第一层
            self.pair_score_mlp[0].weight.zero_()
            self.pair_score_mlp[0].weight[
                :, :self.base_pair_dim
            ].copy_(
                source.pair_score_mlp[0].weight
            )
            self.pair_score_mlp[0].bias.copy_(
                source.pair_score_mlp[0].bias
            )

            # Pair score最后一层
            self.pair_score_mlp[3].load_state_dict(
                source.pair_score_mlp[3].state_dict()
            )

            # Pair feature第一层
            self.pair_feat_mlp[0].weight.zero_()
            self.pair_feat_mlp[0].weight[
                :, :self.base_pair_dim
            ].copy_(
                source.pair_feat_mlp[0].weight
            )
            self.pair_feat_mlp[0].bias.copy_(
                source.pair_feat_mlp[0].bias
            )

            # Pair feature最后一层
            self.pair_feat_mlp[3].load_state_dict(
                source.pair_feat_mlp[3].state_dict()
            )

    def forward(
        self,
        atom_tokens: torch.Tensor,
        atom_batch: torch.Tensor,
        pocket_tokens: torch.Tensor,
        pocket_mask: torch.Tensor,
        rich_atom_features: torch.Tensor,
    ):
        if pocket_tokens.dim() != 3:
            raise ValueError(
                "pocket_tokens must be [B,K,H], "
                f"got {tuple(pocket_tokens.shape)}"
            )

        if (
            pocket_mask.dim() != 2
            or pocket_mask.shape[:2]
            != pocket_tokens.shape[:2]
        ):
            raise ValueError(
                "pocket_mask shape mismatch: "
                f"{tuple(pocket_mask.shape)} vs "
                f"{tuple(pocket_tokens.shape)}"
            )

        if rich_atom_features.dim() != 2:
            raise ValueError(
                "rich_atom_features must be [N_atom,43], "
                f"got {tuple(rich_atom_features.shape)}"
            )

        if (
            rich_atom_features.size(0)
            != atom_tokens.size(0)
        ):
            raise ValueError(
                "rich atom count does not match atom tokens: "
                f"{rich_atom_features.size(0)} vs "
                f"{atom_tokens.size(0)}"
            )

        if (
            rich_atom_features.size(1)
            != self.rich_atom_dim
        ):
            raise ValueError(
                "Unexpected rich atom dimension: "
                f"{rich_atom_features.size(1)}, "
                f"expected {self.rich_atom_dim}"
            )

        batch_size, pocket_k, hidden_dim = (
            pocket_tokens.shape
        )

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Pocket hidden dim={hidden_dim}, "
                f"expected {self.hidden_dim}"
            )

        atom_dense, atom_mask = self._to_dense_tokens(
            atom_tokens,
            atom_batch,
            batch_size,
        )

        rich_dense, rich_mask = self._to_dense_tokens(
            rich_atom_features,
            atom_batch,
            batch_size,
        )

        if not torch.equal(atom_mask, rich_mask):
            raise ValueError(
                "atom token mask and rich atom mask "
                "are not aligned"
            )

        rich_dense = rich_dense.to(
            dtype=atom_dense.dtype
        )

        pocket_key_padding_mask = ~pocket_mask

        # 保留E2原子读取候选残基的Attention
        atom_ctx, _ = self.atom_to_pocket_attn(
            query=atom_dense,
            key=pocket_tokens,
            value=pocket_tokens,
            key_padding_mask=pocket_key_padding_mask,
            need_weights=False,
        )

        atom_enhanced = self.atom_norm(
            atom_dense + self.dropout(atom_ctx)
        )

        atom_exp = atom_enhanced.unsqueeze(2).expand(
            -1,
            -1,
            pocket_k,
            -1,
        )

        pocket_exp = pocket_tokens.unsqueeze(1).expand(
            -1,
            atom_exp.size(1),
            -1,
            -1,
        )

        # 每个原子的43维属性复制到它与K个残基组成的Pair中
        rich_exp = rich_dense.unsqueeze(2).expand(
            -1,
            -1,
            pocket_k,
            -1,
        )

        # E5核心：43维属性直接进入Pair MLP
        pair_input = torch.cat(
            [
                atom_exp,
                pocket_exp,
                atom_exp * pocket_exp,
                torch.abs(atom_exp - pocket_exp),
                rich_exp,
            ],
            dim=-1,
        )

        expected_dim = (
            self.hidden_dim * 4
            + self.rich_atom_dim
        )

        if pair_input.size(-1) != expected_dim:
            raise RuntimeError(
                f"E5 pair dimension mismatch: "
                f"{pair_input.size(-1)} vs {expected_dim}"
            )

        pair_scores = self.pair_score_mlp(
            pair_input
        ).squeeze(-1)

        pair_feat = self.pair_feat_mlp(
            pair_input
        )

        pair_mask = (
            atom_mask.unsqueeze(-1)
            & pocket_mask.unsqueeze(1)
        )

        pair_scores = pair_scores.masked_fill(
            ~pair_mask,
            -1e9,
        )

        pair_alpha = torch.softmax(
            pair_scores.flatten(1),
            dim=-1,
        )

        local_feat = torch.sum(
            pair_alpha.unsqueeze(-1)
            * pair_feat.flatten(1, 2),
            dim=1,
        )

        local_feat = self.out_norm(local_feat)

        return local_feat, pair_scores, atom_mask


class MyModelMDTAP13DFineGrained(E2Model):
    """
    E5模型。

    全局主干严格继承E2；
    只将局部AtomResidueInteraction替换为
    RichAttributeAwareAtomResidueInteraction。
    """

    def __init__(
        self,
        drug_1d_in_dim=768,
        drug_3d_node_in_dim=10,
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=128,
        dropout=0.1,
        task="regression",
        pocket_top_k: int = 64,
        interaction_heads: int = 4,
        rich_atom_in_dim: int = 43,
    ):
        # 先完整初始化E2
        super().__init__(
            drug_1d_in_dim=drug_1d_in_dim,
            drug_3d_node_in_dim=drug_3d_node_in_dim,
            protein_1d_in_dim=protein_1d_in_dim,
            protein_3d_node_s_dim=protein_3d_node_s_dim,
            protein_3d_node_v_dim=protein_3d_node_v_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task,
            pocket_top_k=pocket_top_k,
            interaction_heads=interaction_heads,
        )

        if drug_3d_node_in_dim != 10:
            raise ValueError(
                "E5 keeps the E2 global Drug EGNN at 10D. "
                f"Got drug_3d_node_in_dim="
                f"{drug_3d_node_in_dim}"
            )

        old_interaction = self.atom_residue_interaction

        # 构造新模块时不改变全局RNG状态
        cpu_rng_state = torch.get_rng_state()

        rich_interaction = (
            RichAttributeAwareAtomResidueInteraction(
                hidden_dim=hidden_dim,
                num_heads=interaction_heads,
                dropout=dropout,
                rich_atom_dim=rich_atom_in_dim,
            )
        )

        torch.set_rng_state(cpu_rng_state)

        # 复制E2已有参数，新增43维列保持0
        rich_interaction.copy_e2_parameters(
            old_interaction
        )

        self.atom_residue_interaction = (
            rich_interaction
        )
        self.rich_atom_in_dim = rich_atom_in_dim

    def forward(
        self,
        batch,
        return_debug: bool = False,
    ):
        # ===== Drug global + atom tokens =====
        drug_1d_feat = self.drug_1d_encoder(
            batch["drug_1d"]
        )

        drug_3d_out = self.drug_3d_encoder(
            batch["drug_3d"],
            return_node=True,
        )

        drug_3d_feat = drug_3d_out["graph_feat"]
        drug_atom_tokens = drug_3d_out["node_feat"]
        drug_atom_batch = drug_3d_out["batch"]

        drug_feat = self.drug_fusion(
            [drug_1d_feat, drug_3d_feat]
        )

        rich_atom_raw = batch["drug_3d"].get(
            "rich_x"
        )

        if rich_atom_raw is None:
            raise KeyError(
                "E5 requires batch['drug_3d']['rich_x']. "
                "Use drug_3d_dual10_43."
            )

        if (
            rich_atom_raw.size(0)
            != drug_atom_tokens.size(0)
        ):
            raise ValueError(
                "rich_x and atom tokens are not aligned: "
                f"{rich_atom_raw.size(0)} vs "
                f"{drug_atom_tokens.size(0)}"
            )

        # ===== Protein global + residue tokens =====
        protein_1d_feat = self.protein_1d_encoder(
            batch["protein_1d"]
        )

        protein_3d_out = self.protein_3d_encoder(
            batch["protein_3d"],
            return_node=True,
        )

        protein_3d_feat = protein_3d_out[
            "graph_feat"
        ]
        protein_residue_tokens = protein_3d_out[
            "node_feat"
        ]
        protein_residue_batch = protein_3d_out[
            "batch"
        ]

        protein_feat = self.protein_fusion(
            [protein_1d_feat, protein_3d_feat]
        )

        # ===== E2口袋选择完全保留 =====
        (
            pocket_tokens,
            pocket_mask,
            pocket_scores,
            pocket_indices,
        ) = self.pocket_selector(
            protein_tokens=protein_residue_tokens,
            protein_batch=protein_residue_batch,
            drug_context=drug_feat,
        )

        # ===== E5 Rich-aware Pair Interaction =====
        (
            local_interaction_feat,
            interaction_scores,
            atom_mask,
        ) = self.atom_residue_interaction(
            atom_tokens=drug_atom_tokens,
            atom_batch=drug_atom_batch,
            pocket_tokens=pocket_tokens,
            pocket_mask=pocket_mask,
            rich_atom_features=rich_atom_raw,
        )

        # ===== E2 Baseline主分支 =====
        base_pair_feat = torch.cat(
            [drug_feat, protein_feat],
            dim=-1,
        )

        base_pred = self.decoder(
            base_pair_feat
        )

        # ===== E2局部残差分支 =====
        residual_pair_feat = torch.cat(
            [
                drug_feat,
                protein_feat,
                local_interaction_feat,
            ],
            dim=-1,
        )

        local_delta = self.local_delta_head(
            residual_pair_feat
        )

        local_delta = local_delta.view_as(
            base_pred
        )

        out = base_pred + local_delta

        if not return_debug:
            return out

        return {
            "pred": out,
            "base_pred": base_pred,
            "local_delta": local_delta,
            "drug_feat": drug_feat,
            "protein_feat": protein_feat,
            "local_interaction_feat": (
                local_interaction_feat
            ),
            "drug_atom_tokens": drug_atom_tokens,
            "rich_atom_raw": rich_atom_raw,
            "drug_atom_batch": drug_atom_batch,
            "protein_residue_tokens": (
                protein_residue_tokens
            ),
            "protein_residue_batch": (
                protein_residue_batch
            ),
            "pocket_tokens": pocket_tokens,
            "pocket_mask": pocket_mask,
            "pocket_scores": pocket_scores,
            "pocket_indices": pocket_indices,
            "interaction_scores": interaction_scores,
            "atom_mask": atom_mask,
        }
