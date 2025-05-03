import torch
from torch import nn

class PrefixMapper(nn.Module):
    def __init__(self, context_dim, target_dim, hidden_dim=512, prefix_len=1):
        super().__init__()
        self.prefix_len = prefix_len
        self.mapper = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim * prefix_len)
        )
        self.target_dim = target_dim

    def forward(self, context_vec):
        mapped = self.mapper(context_vec)  # (B, target_dim * prefix_len)
        return mapped.view(-1, self.prefix_len, self.target_dim)  # (B, prefix_len, dim)
    
    @staticmethod
    def load(file_path):
        model = torch.load(file_path, weights_only=True)
        return model

