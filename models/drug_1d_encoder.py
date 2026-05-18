import torch
import torch.nn as nn


class Drug1DEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128):  # 注意这里
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x)