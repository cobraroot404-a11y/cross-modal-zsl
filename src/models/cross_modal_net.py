"""Full cross-modal alignment network for zero-shot image classification.

Pipeline (mirrors the report's proposed system):
  1. Transfer-learning CNN backbone extracts a spatial visual feature map.
  2. Attribute-guided attention produces, for every candidate class, a
     "response" vector describing how strongly the image reacts to that
     class's textual query.
  3. A semantic encoder maps each class's attribute vector into the same
     embedding space.
  4. Cosine similarity between the response and the class's own semantic
     embedding gives the matching score used for classification -- since
     step 2-4 only need a class's attribute vector (never its images), the
     same model classifies unseen classes at test time without retraining.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttributeGuidedAttention
from .backbone import ResNetBackbone
from .semantic_encoder import SemanticEncoder


class CrossModalZSLNet(nn.Module):
    def __init__(
        self,
        attribute_dim: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone_stages: int = 6,
        embed_dim: int = 512,
        attribute_hidden_dim: int = 512,
    ):
        super().__init__()
        self.backbone = ResNetBackbone(backbone, pretrained=pretrained, freeze_stages=freeze_backbone_stages)
        self.attention = AttributeGuidedAttention(self.backbone.out_channels, embed_dim)
        self.semantic_encoder = SemanticEncoder(attribute_dim, attribute_hidden_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(
        self, images: torch.Tensor, class_attributes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        images:           (B, 3, H, W)
        class_attributes: (C, attribute_dim) -- candidate classes for this forward pass

        Returns:
            scores:   (B, C) cosine-similarity matching scores
            attn_map: (B, C, h, w) response/attention maps
        """
        feature_map = self.backbone(images)                              # (B, Cv, h, w)
        semantic_embeds = self.semantic_encoder(class_attributes)        # (C, D)
        attended, attn_map = self.attention(feature_map, semantic_embeds)  # (B, C, D)

        attended_n = F.normalize(attended, dim=-1)
        semantic_n = F.normalize(semantic_embeds, dim=-1)
        scores = torch.einsum("bcd,cd->bc", attended_n, semantic_n)      # cosine similarity
        return scores, attn_map

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, class_attributes: torch.Tensor, class_ids
    ):
        """Convenience inference helper -> (predicted_class_ids, probs, attn_map)."""
        self.eval()
        scores, attn_map = self.forward(images, class_attributes)
        probs = F.softmax(scores / 0.1, dim=-1)
        top_idx = probs.argmax(dim=-1)
        predicted = [class_ids[i] for i in top_idx.tolist()]
        return predicted, probs, attn_map
