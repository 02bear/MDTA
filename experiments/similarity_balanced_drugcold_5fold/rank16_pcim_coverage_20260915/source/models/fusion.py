import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """
    将多个模态特征拼接后投影到统一维度
    """
    def __init__(self, input_dims, out_dim=256, hidden_dim=512, dropout=0.1):
        super().__init__()
        total_dim = sum(input_dims)

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, features):
        """
        features: list[Tensor], 每个 Tensor 形状为 [B, Di]
        """
        x = torch.cat(features, dim=-1)
        return self.fusion(x)