import torch.nn as nn


class Decoder(nn.Module):
    """
    通用解码器
    task = 'classification' 或 'regression'
    """
    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.1, task="classification"):
        super().__init__()
        self.task = task

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        out = self.mlp(x)
        return out