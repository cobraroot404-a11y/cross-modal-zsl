"""Attribute-guided cross-modal attention.

Implements the "response map" idea from the report's existing-system
description: each class's text-attribute embedding acts as a query that is
compared against every spatial location of the visual feature map, producing
an attention (response) map. The attended visual feature is then compared to
the same attribute embedding via cosine similarity for classification.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeGuidedAttention(nn.Module):
    def __init__(self, visual_channels: int, embed_dim: int):
        super().__init__()
        self.visual_proj = nn.Conv2d(visual_channels, embed_dim, kernel_size=1)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.scale = embed_dim ** -0.5

    def forward(
        self, feature_map: torch.Tensor, semantic_embeds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feature_map:     (B, Cv, H, W)  raw backbone spatial features
        semantic_embeds: (C, D)         one embedding per candidate class

        Returns:
            attended: (B, C, D) attention-weighted visual response per class query
            attn_map: (B, C, H, W) the response/attention map for visualization
        """
        b, _, h, w = feature_map.shape
        f = self.visual_proj(feature_map)          # (B, D, H, W)
        k = self.key_proj(f).flatten(2)             # (B, D, N)
        v = self.value_proj(f).flatten(2)           # (B, D, N)
        q = self.query_proj(semantic_embeds)        # (C, D)

        # (B, C, N) = einsum over shared D between per-image keys and per-class queries
        attn_logits = torch.einsum("cd,bdn->bcn", q, k) * self.scale
        attn = F.softmax(attn_logits, dim=-1)       # (B, C, N)

        attended = torch.einsum("bcn,bdn->bcd", attn, v)  # (B, C, D)
        attn_map = attn.view(b, semantic_embeds.size(0), h, w)
        return attended, attn_map
