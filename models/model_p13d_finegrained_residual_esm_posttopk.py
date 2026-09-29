# -*- coding: utf-8 -*-
"""
Baseline + fine-grained atom-residue local interaction.

设计目标：
1. 保留原 baseline 全局分支：
   drug_1d + drug_3d -> drug_feat
   protein_1d + protein_3d -> protein_feat
2. 额外从 3D encoder 取局部节点特征：
   drug_atom_tokens、protein_residue_tokens
3. 使用 drug-conditioned pocket selector 从全蛋白残基中选 top-K 候选 pocket tokens；
4. 使用 atom-residue interaction 模块得到 local_interaction_feat；
5. 最终预测头输入从 [drug_feat, protein_feat] 扩展为：
   [drug_feat, protein_feat, local_interaction_feat]

注意：
- 不引入 PCL / SoftCLIP / I2MoE / 额外 loss；
- 训练脚本仍然只需要 pred = model(batch)，然后用 MSELoss；
- 依赖 Drug3DEGNNEncoder 和 Protein3DEGNNEncoder 支持 return_node=True。
"""

import torch
import torch.nn as nn

from models.drug_1d_encoder import Drug1DEncoder
from models.drug_3d_egnn_encoder import Drug3DEGNNEncoder
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_egnn_encoder import Protein3DEGNNEncoder
from models.fusion import ConcatFusion
from models.decoder import Decoder


class DrugConditionedPocketSelector(nn.Module):
    """
    药物条件候选口袋选择器。

    输入：
        protein_tokens: [N_res_total, H]  所有蛋白残基级特征
        protein_batch:  [N_res_total]     每个残基属于 batch 中哪个 protein
        drug_context:   [B, H]            当前 drug-protein pair 的药物全局特征

    输出：
        pocket_tokens:  [B, top_k, H]     每个蛋白选出的候选 pocket 残基特征
        pocket_mask:    [B, top_k]        True 表示有效 token，False 表示 padding
        pocket_scores:  [B, top_k]        top-K 残基的选择分数；padding 为 -inf
        pocket_indices: [B, top_k]        top-K 残基在 protein_tokens 中的全局索引；padding 为 -1
    """

    def __init__(self, hidden_dim: int, top_k: int = 64, dropout: float = 0.1):
        super().__init__()
        if top_k <= 0:
            raise ValueError(f"top_k 必须为正数，实际 top_k={top_k}")

        self.hidden_dim = hidden_dim
        self.top_k = top_k

        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        protein_tokens: torch.Tensor,
        protein_batch: torch.Tensor,
        drug_context: torch.Tensor,
    ):
        if protein_tokens.dim() != 2:
            raise ValueError(f"protein_tokens 应为 [N,H]，实际 {tuple(protein_tokens.shape)}")
        if protein_batch.dim() != 1 or protein_batch.size(0) != protein_tokens.size(0):
            raise ValueError(
                f"protein_batch 应为 [N] 且 N 与 protein_tokens 一致，"
                f"实际 protein_batch={tuple(protein_batch.shape)}, protein_tokens={tuple(protein_tokens.shape)}"
            )
        if drug_context.dim() != 2 or drug_context.size(1) != protein_tokens.size(1):
            raise ValueError(
                f"drug_context 应为 [B,H] 且 H 与 protein_tokens 一致，"
                f"实际 drug_context={tuple(drug_context.shape)}, protein_tokens={tuple(protein_tokens.shape)}"
            )

        batch_size = drug_context.size(0)
        hidden_dim = protein_tokens.size(1)
        device = protein_tokens.device
        dtype = protein_tokens.dtype

        if protein_batch.numel() > 0:
            max_batch_id = int(protein_batch.max().item())
            if max_batch_id >= batch_size:
                raise ValueError(
                    f"protein_batch 中最大图编号 {max_batch_id} 超过 drug_context batch_size={batch_size}"
                )

        # 每个残基拼接当前 pair 的 drug context，形成 drug-conditioned residue score。
        drug_per_residue = drug_context[protein_batch]
        score_input = torch.cat(
            [
                protein_tokens,
                drug_per_residue,
                protein_tokens * drug_per_residue,
                torch.abs(protein_tokens - drug_per_residue),
            ],
            dim=-1,
        )
        residue_scores = self.score_mlp(score_input).squeeze(-1)  # [N_res_total]

        pocket_tokens = torch.zeros(batch_size, self.top_k, hidden_dim, device=device, dtype=dtype)
        pocket_mask = torch.zeros(batch_size, self.top_k, device=device, dtype=torch.bool)
        pocket_scores = torch.full(
            (batch_size, self.top_k),
            fill_value=-torch.inf,
            device=device,
            dtype=residue_scores.dtype,
        )
        pocket_indices = torch.full(
            (batch_size, self.top_k),
            fill_value=-1,
            device=device,
            dtype=torch.long,
        )

        # 按 batch 内每个蛋白逐个 top-k。Davis/KIBA 常用 batch_size 不大，这个循环稳定直观。
        for b in range(batch_size):
            idx = torch.nonzero(protein_batch == b, as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue

            k = min(self.top_k, idx.numel())
            scores_b = residue_scores[idx]
            top_scores, top_pos = torch.topk(scores_b, k=k, largest=True, sorted=True)
            selected_idx = idx[top_pos]

            # 取出当前硬 Top-K 选择的残基特征。
            selected_tokens = protein_tokens[selected_idx]

            # 对当前 Top-K 分数进行样本内标准化。
            # 避免原始分数尺度不断变大，使门控函数迅速饱和。
            score_mean = top_scores.mean()
            score_std = top_scores.std(unbiased=False).clamp_min(1e-6)
            normalized_scores = (top_scores - score_mean) / score_std

            # 使用有界平滑门控代替全局 softmax。
            # 每个残基权重限制在 [0.5, 1.5]，平均尺度仍接近 1。
            # 不再让 64 个残基相互竞争，也不会出现单个权重接近 64。
            selection_weights = 1.0 + 0.5 * torch.tanh(normalized_scores)

            weighted_tokens = selected_tokens * selection_weights.unsqueeze(-1)

            pocket_tokens[b, :k] = weighted_tokens
            pocket_mask[b, :k] = True
            pocket_scores[b, :k] = top_scores
            pocket_indices[b, :k] = selected_idx

        return pocket_tokens, pocket_mask, pocket_scores, pocket_indices


class AtomResidueInteraction(nn.Module):
    """
    药物原子 - 候选口袋残基局部交互模块。

    输入：
        atom_tokens:   [N_atom_total, H]
        atom_batch:    [N_atom_total]
        pocket_tokens: [B, K, H]
        pocket_mask:   [B, K]，True 表示有效 pocket token

    输出：
        local_interaction_feat: [B, H]
        interaction_scores:     [B, A_max, K]
        atom_mask:              [B, A_max]
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads 必须为正数，实际 num_heads={num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim 必须能被 num_heads 整除，实际 hidden_dim={hidden_dim}, num_heads={num_heads}"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.atom_to_pocket_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.atom_norm = nn.LayerNorm(hidden_dim)

        self.pair_score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.pair_feat_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _to_dense_tokens(tokens: torch.Tensor, batch: torch.Tensor, batch_size: int):
        """
        将 PyG 风格的拼接节点表示 [N,H] 转成 dense 表示 [B,A_max,H]。
        """
        if tokens.dim() != 2:
            raise ValueError(f"tokens 应为 [N,H]，实际 {tuple(tokens.shape)}")
        if batch.dim() != 1 or batch.size(0) != tokens.size(0):
            raise ValueError(
                f"batch 应为 [N] 且 N 与 tokens 一致，实际 batch={tuple(batch.shape)}, tokens={tuple(tokens.shape)}"
            )

        device = tokens.device
        dtype = tokens.dtype
        hidden_dim = tokens.size(1)

        if batch.numel() == 0:
            dense = torch.zeros(batch_size, 1, hidden_dim, device=device, dtype=dtype)
            mask = torch.zeros(batch_size, 1, device=device, dtype=torch.bool)
            return dense, mask

        max_batch_id = int(batch.max().item())
        if max_batch_id >= batch_size:
            raise ValueError(f"batch 中最大图编号 {max_batch_id} 超过 batch_size={batch_size}")

        counts = torch.bincount(batch, minlength=batch_size)
        max_len = int(counts.max().item()) if counts.numel() > 0 else 1
        max_len = max(max_len, 1)

        dense = torch.zeros(batch_size, max_len, hidden_dim, device=device, dtype=dtype)
        mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.bool)

        for b in range(batch_size):
            idx = torch.nonzero(batch == b, as_tuple=False).view(-1)
            n = idx.numel()
            if n == 0:
                continue
            dense[b, :n] = tokens[idx]
            mask[b, :n] = True

        return dense, mask

    def forward(
        self,
        atom_tokens: torch.Tensor,
        atom_batch: torch.Tensor,
        pocket_tokens: torch.Tensor,
        pocket_mask: torch.Tensor,
    ):
        if pocket_tokens.dim() != 3:
            raise ValueError(f"pocket_tokens 应为 [B,K,H]，实际 {tuple(pocket_tokens.shape)}")
        if pocket_mask.dim() != 2 or pocket_mask.shape[:2] != pocket_tokens.shape[:2]:
            raise ValueError(
                f"pocket_mask 应为 [B,K] 且与 pocket_tokens 前两维一致，"
                f"实际 pocket_mask={tuple(pocket_mask.shape)}, pocket_tokens={tuple(pocket_tokens.shape)}"
            )

        batch_size, pocket_k, hidden_dim = pocket_tokens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"pocket hidden_dim={hidden_dim} 与模块 hidden_dim={self.hidden_dim} 不一致")

        atom_dense, atom_mask = self._to_dense_tokens(atom_tokens, atom_batch, batch_size)
        if atom_dense.size(-1) != self.hidden_dim:
            raise ValueError(f"atom hidden_dim={atom_dense.size(-1)} 与模块 hidden_dim={self.hidden_dim} 不一致")

        # MultiheadAttention 的 key_padding_mask: True 表示该位置需要被忽略。
        pocket_key_padding_mask = ~pocket_mask

        # 每个药物原子读取候选 pocket 残基信息。
        atom_ctx, _ = self.atom_to_pocket_attn(
            query=atom_dense,
            key=pocket_tokens,
            value=pocket_tokens,
            key_padding_mask=pocket_key_padding_mask,
            need_weights=False,
        )
        atom_enhanced = self.atom_norm(atom_dense + self.dropout(atom_ctx))  # [B,A,H]

        # 构建显式 atom-residue pair 表示。
        atom_exp = atom_enhanced.unsqueeze(2).expand(-1, -1, pocket_k, -1)      # [B,A,K,H]
        pocket_exp = pocket_tokens.unsqueeze(1).expand(-1, atom_exp.size(1), -1, -1)  # [B,A,K,H]
        pair_input = torch.cat(
            [
                atom_exp,
                pocket_exp,
                atom_exp * pocket_exp,
                torch.abs(atom_exp - pocket_exp),
            ],
            dim=-1,
        )  # [B,A,K,4H]

        pair_scores = self.pair_score_mlp(pair_input).squeeze(-1)  # [B,A,K]
        pair_feat = self.pair_feat_mlp(pair_input)                 # [B,A,K,H]

        pair_mask = atom_mask.unsqueeze(-1) & pocket_mask.unsqueeze(1)  # [B,A,K]
        pair_scores = pair_scores.masked_fill(~pair_mask, -1e9)

        # 对有效 atom-residue pair 做加权池化。无效 pair 的 softmax 权重接近 0。
        pair_alpha = torch.softmax(pair_scores.flatten(1), dim=-1)  # [B,A*K]
        local_feat = torch.sum(
            pair_alpha.unsqueeze(-1) * pair_feat.flatten(1, 2),
            dim=1,
        )  # [B,H]
        local_feat = self.out_norm(local_feat)

        return local_feat, pair_scores, atom_mask


class MyModelMDTAP13DFineGrained(nn.Module):
    """
    Baseline + fine-grained atom-residue local interaction。

    全局分支保持 baseline 逻辑：
        drug_1d + drug_3d -> drug_feat
        protein_1d + protein_3d -> protein_feat

    新增局部分支：
        drug_atom_tokens x selected_pocket_tokens -> local_interaction_feat

    最终：
        [drug_feat, protein_feat, local_interaction_feat] -> Decoder
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
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pocket_top_k = pocket_top_k
        self.interaction_heads = interaction_heads

        self.drug_1d_encoder = Drug1DEncoder(input_dim=drug_1d_in_dim, hidden_dim=hidden_dim)
        self.drug_3d_encoder = Drug3DEGNNEncoder(
            node_in_dim=drug_3d_node_in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=3,
            dropout=dropout,
        )
        self.drug_fusion = ConcatFusion(
            input_dims=[hidden_dim, hidden_dim],
            out_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            dropout=dropout,
        )

        self.protein_1d_encoder = Protein1DEncoder(input_dim=protein_1d_in_dim, hidden_dim=hidden_dim)
        self.protein_3d_encoder = Protein3DEGNNEncoder(
            node_s_dim=protein_3d_node_s_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
            n_layers=3,
        )
        self.protein_fusion = ConcatFusion(
            input_dims=[hidden_dim, hidden_dim],
            out_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            dropout=dropout,
        )

        self.pocket_selector = DrugConditionedPocketSelector(
            hidden_dim=hidden_dim,
            top_k=pocket_top_k,
            dropout=dropout,
        )
        self.atom_residue_interaction = AtomResidueInteraction(
            hidden_dim=hidden_dim,
            num_heads=interaction_heads,
            dropout=dropout,
        )

        # ============================================================
        # E2: Baseline主预测分支
        # 只使用drug_feat和protein_feat，与原始Baseline保持一致。
        # 继续使用self.decoder这个名称，兼容现有训练脚本。
        # ============================================================
        self.decoder = Decoder(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task,
        )

        # ============================================================
        # E2: 细粒度局部残差分支
        # 输入全局药物、全局蛋白和局部原子-残基交互特征，
        # 只预测对Baseline结果的修正量local_delta。
        # ============================================================
        self.local_delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # 最后一层零初始化：
        # 训练开始时local_delta严格为0，
        # 因而模型初始行为等价于Baseline主分支。
        nn.init.zeros_(self.local_delta_head[-1].weight)
        nn.init.zeros_(self.local_delta_head[-1].bias)

        # ============================================================
        # E3.1: Top-K之后的残基级ESM残差Adapter
        #
        # 放在所有E2模块之后初始化，避免改变E2原模块的
        # 随机初始化顺序。
        # ============================================================
        self.pocket_esm_adapter = nn.Sequential(
            nn.Linear(protein_1d_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 初始时ESM增量严格为0，模型从E2行为开始训练。
        nn.init.zeros_(
            self.pocket_esm_adapter[-1].weight
        )
        nn.init.zeros_(
            self.pocket_esm_adapter[-1].bias
        )

        # 控制ESM局部残差幅度，避免高维ESM表示过强干扰。
        self.esm_residual_scale = 0.1

    def forward(self, batch, return_debug: bool = False):
        # ===== Drug global + atom tokens =====
        drug_1d_feat = self.drug_1d_encoder(batch["drug_1d"])  # [B,H]
        drug_3d_out = self.drug_3d_encoder(batch["drug_3d"], return_node=True)
        drug_3d_feat = drug_3d_out["graph_feat"]                # [B,H]
        drug_atom_tokens = drug_3d_out["node_feat"]             # [N_atom_total,H]
        drug_atom_batch = drug_3d_out["batch"]                  # [N_atom_total]
        drug_feat = self.drug_fusion([drug_1d_feat, drug_3d_feat])

        # ===== Protein global + residue tokens =====
        protein_1d_feat = self.protein_1d_encoder(batch["protein_1d"])  # [B,H]
        protein_3d_out = self.protein_3d_encoder(batch["protein_3d"], return_node=True)
        protein_3d_feat = protein_3d_out["graph_feat"]                  # [B,H]
        protein_residue_tokens = protein_3d_out["node_feat"]            # [N_res_total,H]
        protein_residue_batch = protein_3d_out["batch"]                 # [N_res_total]
        protein_feat = self.protein_fusion([protein_1d_feat, protein_3d_feat])

        # ===== Drug-conditioned pocket selection =====
        # ============================================================
        # E3.1 Step 1:
        # 口袋选择仍然完全使用E2的3D残基特征。
        # ESM不会改变Top-K排序。
        # ============================================================
        pocket_tokens_3d, pocket_mask, pocket_scores, pocket_indices = (
            self.pocket_selector(
                protein_tokens=protein_residue_tokens,
                protein_batch=protein_residue_batch,
                drug_context=drug_feat,
            )
        )

        # ============================================================
        # E3.1 Step 2:
        # 完整ESM以长度为B的CPU列表保存，不再进行全batch拼接。
        # 根据Top-K全局索引转换为每条蛋白内部的局部索引，
        # 只抽取B × K个残基的ESM表示。
        # ============================================================
        residue_esm_list = batch[
            "protein_residue_1d_list"
        ]
        residue_esm_mask_list = batch[
            "protein_residue_1d_mask_list"
        ]

        batch_size = pocket_indices.size(0)

        if len(residue_esm_list) != batch_size:
            raise ValueError(
                "ESM list batch size mismatch: "
                f"{len(residue_esm_list)} vs {batch_size}"
            )

        if len(residue_esm_mask_list) != batch_size:
            raise ValueError(
                "ESM mask list batch size mismatch: "
                f"{len(residue_esm_mask_list)} vs {batch_size}"
            )

        selected_esm_chunks = []
        selected_mask_chunks = []

        # protein_residue_tokens是按batch顺序拼接的，
        # 因此running_offset表示当前蛋白在全局节点中的起点。
        running_offset = 0

        for b in range(batch_size):
            esm_b = residue_esm_list[b]
            esm_mask_b = residue_esm_mask_list[b].bool()

            if esm_b.device.type != "cpu":
                raise RuntimeError(
                    "Full residue ESM must remain on CPU."
                )

            num_residues = int(esm_b.size(0))

            if num_residues <= 0:
                raise ValueError(
                    f"Protein {b} has no residue ESM tokens."
                )

            global_indices_cpu = (
                pocket_indices[b]
                .detach()
                .cpu()
            )

            valid_pocket_cpu = (
                pocket_mask[b]
                .detach()
                .cpu()
            )

            local_indices_cpu = (
                global_indices_cpu - running_offset
            )

            if valid_pocket_cpu.any():
                valid_local_indices = local_indices_cpu[
                    valid_pocket_cpu
                ]

                min_idx = int(valid_local_indices.min().item())
                max_idx = int(valid_local_indices.max().item())

                if min_idx < 0 or max_idx >= num_residues:
                    raise IndexError(
                        f"Top-K local index out of range at batch {b}: "
                        f"min={min_idx}, max={max_idx}, "
                        f"num_residues={num_residues}, "
                        f"offset={running_offset}"
                    )

            safe_local_indices = local_indices_cpu.clamp(
                min=0,
                max=num_residues - 1,
            )

            selected_esm_b = esm_b[
                safe_local_indices
            ]

            selected_mask_b = (
                esm_mask_b[safe_local_indices]
                & valid_pocket_cpu
            )

            selected_esm_chunks.append(
                selected_esm_b
            )
            selected_mask_chunks.append(
                selected_mask_b
            )

            running_offset += num_residues

        # 此时只拼接B × K × 1280，而不是所有残基。
        selected_esm_raw_cpu = torch.stack(
            selected_esm_chunks,
            dim=0,
        ).contiguous()

        selected_esm_mask_cpu = torch.stack(
            selected_mask_chunks,
            dim=0,
        ).contiguous()

        selected_esm_raw = selected_esm_raw_cpu.to(
            device=pocket_tokens_3d.device,
            dtype=pocket_tokens_3d.dtype,
        )

        selected_esm_mask = selected_esm_mask_cpu.to(
            device=pocket_mask.device,
        )

        selected_esm_raw = (
            selected_esm_raw
            * selected_esm_mask.unsqueeze(-1).to(
                dtype=selected_esm_raw.dtype
            )
        )

        # ============================================================
        # E3.1 Step 3:
        # 仅投影B × K个已选残基，而不是投影所有残基。
        # ============================================================
        pocket_esm_delta = self.pocket_esm_adapter(
            selected_esm_raw
        )

        pocket_esm_delta = (
            pocket_esm_delta
            * selected_esm_mask.unsqueeze(-1).to(
                dtype=pocket_esm_delta.dtype
            )
        )

        # Top-K 3D口袋作为主体，ESM只提供受控残差补充。
        pocket_tokens = (
            pocket_tokens_3d
            + self.esm_residual_scale
            * pocket_esm_delta
        )

        # ===== Atom-residue local interaction =====
        local_interaction_feat, interaction_scores, atom_mask = self.atom_residue_interaction(
            atom_tokens=drug_atom_tokens,
            atom_batch=drug_atom_batch,
            pocket_tokens=pocket_tokens,
            pocket_mask=pocket_mask,
        )

        # ============================================================
        # E2: Baseline主分支
        # ============================================================
        base_pair_feat = torch.cat(
            [drug_feat, protein_feat],
            dim=-1,
        )
        base_pred = self.decoder(base_pair_feat)

        # ============================================================
        # E2: 细粒度局部残差分支
        # ============================================================
        residual_pair_feat = torch.cat(
            [drug_feat, protein_feat, local_interaction_feat],
            dim=-1,
        )
        local_delta = self.local_delta_head(residual_pair_feat)

        # 兼容Decoder输出为[B]或者[B,1]的情况。
        local_delta = local_delta.view_as(base_pred)

        # 最终预测 = Baseline预测 + 局部修正量
        out = base_pred + local_delta

        if not return_debug:
            return out

        return {
            "pred": out,
            "base_pred": base_pred,
            "local_delta": local_delta,
            "drug_feat": drug_feat,
            "protein_feat": protein_feat,
            "local_interaction_feat": local_interaction_feat,
            "drug_atom_tokens": drug_atom_tokens,
            "drug_atom_batch": drug_atom_batch,
            "protein_residue_tokens": protein_residue_tokens,
            "protein_residue_batch": protein_residue_batch,
            "pocket_tokens": pocket_tokens,
            "pocket_tokens_3d": pocket_tokens_3d,
            "pocket_esm_delta": pocket_esm_delta,
            "selected_esm_mask": selected_esm_mask,
            "pocket_mask": pocket_mask,
            "pocket_scores": pocket_scores,
            "pocket_indices": pocket_indices,
            "interaction_scores": interaction_scores,
            "atom_mask": atom_mask,
        }
