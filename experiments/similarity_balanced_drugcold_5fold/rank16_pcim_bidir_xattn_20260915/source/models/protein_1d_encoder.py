import torch.nn as nn


class Protein1DEncoder(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=128):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x)