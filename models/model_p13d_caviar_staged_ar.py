"""Two-stage CAVIAR model with an exactly protected global predictor.

Stage 1 trains only the global affinity path.  Stage 2 keeps every global
module frozen and in evaluation mode, then trains only CAVIAR subpocket
encoding and the atom-residue (AR) residual correction.  The FP path from SP1
is deliberately excluded so this experiment isolates whether subpocket AR
evidence generalizes beyond the validation drugs.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from models.model_p13d_caviar_subpocket import CaviarSubpocketDTA


class CaviarStagedARDTA(CaviarSubpocketDTA):
    GLOBAL_MODULE_NAMES = (
        "drug_1d_encoder",
        "drug_atom_encoder",
        "drug_fusion",
        "protein_1d_encoder",
        "protein_3d_encoder",
        "residue_type_embedding",
        "residue_aux_proj",
        "residue_norm",
        "protein_fusion",
        "global_decoder",
    )
    AR_MODULE_NAMES = (
        "subpocket_gcn",
        "subpocket_graph",
        "region_builder",
        "ar_pool_score",
        "ar_gate",
        "ar_delta_head",
    )
    UNUSED_FP_MODULE_NAMES = ("region_propagation", "fp_gate", "fp_delta_head")

    def named_stage_modules(self, names):
        for name in names:
            yield name, getattr(self, name)

    def stage_parameters(self, stage):
        names = self.GLOBAL_MODULE_NAMES if stage == "global" else self.AR_MODULE_NAMES
        for _, module in self.named_stage_modules(names):
            yield from module.parameters()

    def configure_stage(self, stage, training):
        if stage not in {"global", "ar"}:
            raise ValueError(f"Unknown stage: {stage}")
        self.train(training)
        active_names = (
            set(self.GLOBAL_MODULE_NAMES)
            if stage == "global"
            else set(self.AR_MODULE_NAMES)
        )
        for name, parameter in self.named_parameters():
            module_name = name.split(".", 1)[0]
            parameter.requires_grad_(module_name in active_names)
        if stage == "ar":
            # Frozen parameters alone are insufficient: dropout in train mode
            # would otherwise make the supposedly fixed base predictor drift.
            for _, module in self.named_stage_modules(self.GLOBAL_MODULE_NAMES):
                module.eval()
            for _, module in self.named_stage_modules(self.UNUSED_FP_MODULE_NAMES):
                module.eval()

    def encode_global(self, batch):
        drug_1d = self.drug_1d_encoder(batch["drug_1d"])
        drug_out = self.drug_atom_encoder(batch["drug_atom_v2"], return_node=True)
        drug_feat = self.drug_fusion([drug_1d, drug_out["graph_feat"]])

        protein_1d = self.protein_1d_encoder(batch["protein_1d"])
        protein_out = self.protein_3d_encoder(batch["protein_3d"], return_node=True)
        residue = batch["protein_residue_v2"]
        residue_aux = torch.cat(
            [
                self.residue_type_embedding(
                    residue["residue_type_index"].clamp(0, 31)
                ),
                residue["sequence_scalar3"],
                residue["esm_per_tok"],
                residue["esm_mask"].float().unsqueeze(-1),
                residue["structure_extra5"],
            ],
            dim=-1,
        )
        residue_tokens = self.residue_norm(
            protein_out["node_feat"] + self.residue_aux_proj(residue_aux)
        )
        protein_local_global = global_mean_pool(
            residue_tokens, protein_out["batch"]
        )
        protein_feat = self.protein_fusion(
            [protein_1d, protein_out["graph_feat"] + protein_local_global]
        )
        base_pred = self.global_decoder(
            torch.cat([drug_feat, protein_feat], dim=-1)
        )
        return {
            "base_pred": base_pred,
            "drug_feat": drug_feat,
            "protein_feat": protein_feat,
            "atom_tokens": drug_out["node_feat"],
            "residue_tokens": residue_tokens,
        }

    def forward_stage(self, batch, stage):
        encoded = self.encode_global(batch)
        base_pred = encoded["base_pred"]
        if stage == "global":
            return {"pred": base_pred, "base_pred": base_pred}

        pocket_residue, pocket_tokens, pocket_residue_weight = self.subpocket_gcn(
            encoded["residue_tokens"], batch
        )
        pocket_tokens = self.subpocket_graph(pocket_tokens, batch)
        regions, region_mask, region_strength, selected, selected_score = (
            self.region_builder(
                encoded["atom_tokens"],
                pocket_residue,
                pocket_tokens,
                batch,
            )
        )
        ar_score = self.ar_pool_score(regions).squeeze(-1).masked_fill(
            ~region_mask, -1e9
        )
        ar_weight = torch.softmax(ar_score, dim=-1)
        ar_weight = ar_weight * region_mask.to(ar_weight.dtype)
        ar_weight = ar_weight / ar_weight.sum(-1, keepdim=True).clamp_min(1e-8)
        ar_feat = (ar_weight[:, :, None] * regions).sum(1)

        ar_features = torch.cat(
            [encoded["drug_feat"], encoded["protein_feat"], ar_feat], dim=-1
        )
        ar_delta_raw = self.ar_delta_head(ar_features).view_as(base_pred)
        ar_delta = self.delta_max * torch.tanh(ar_delta_raw)
        ar_gate_logit = self.ar_gate(ar_features)
        ar_gate = self.ar_gate_epsilon + (
            1.0 - 2.0 * self.ar_gate_epsilon
        ) * torch.sigmoid(ar_gate_logit)
        pred = base_pred + ar_gate.view_as(base_pred) * ar_delta
        return {
            "pred": pred,
            "base_pred": base_pred,
            "ar_candidate": base_pred + ar_delta,
            "ar_delta": ar_delta,
            "ar_delta_raw": ar_delta_raw,
            "ar_gate": ar_gate,
            "ar_gate_logit": ar_gate_logit,
            "region_mask": region_mask,
            "region_strength": region_strength,
            "ar_region_weight": ar_weight,
            "selected_subpockets": selected,
            "selection_score": selected_score,
            "subpocket_residue_weight": pocket_residue_weight,
        }

    @torch.no_grad()
    def verify_protected_base(self, batch, reference):
        current = self.forward_stage(batch, "global")["base_pred"].float()
        return {
            "max_abs_difference": float((current - reference.float()).abs().max()),
            "allclose": bool(torch.allclose(current, reference.float(), atol=1e-7, rtol=1e-6)),
        }
