"""Training losses.

Total loss = temperature-scaled cross-entropy over cosine-similarity scores
             + a semantic-consistency term that explicitly pulls the
               attended visual response of the ground-truth class towards
               its own text-attribute embedding (the "novel loss function
               that encourages semantic consistency" mentioned in the
               report's abstract).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAlignmentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, consistency_weight: float = 0.5, label_smoothing: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.consistency_weight = consistency_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        scores: (B, C) cosine similarities in [-1, 1]
        labels: (B,)   index of ground-truth class within the C candidates
        """
        logits = scores / self.temperature
        ce_loss = self.ce(logits, labels)

        gt_scores = scores.gather(1, labels.unsqueeze(1)).squeeze(1)  # (B,)
        consistency_loss = (1.0 - gt_scores).mean()

        total = ce_loss + self.consistency_weight * consistency_loss
        return {
            "loss": total,
            "ce_loss": ce_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
        }
