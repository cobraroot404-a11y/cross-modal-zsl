"""Maps class-level text/attribute vectors into the shared embedding space."""
import torch
import torch.nn as nn


class SemanticEncoder(nn.Module):
    def __init__(self, attribute_dim: int, hidden_dim: int = 512, embed_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(attribute_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, attributes: torch.Tensor) -> torch.Tensor:
        """attributes: (C, attribute_dim) -> (C, embed_dim)"""
        return self.net(attributes)
