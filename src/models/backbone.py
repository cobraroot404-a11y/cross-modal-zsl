"""Transfer-learning visual backbone (report Module 3).

Wraps a pretrained torchvision ResNet, exposing the spatial feature map
(before global pooling) so the attention module can attend over locations,
as described in the report's "response map" formulation.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm

_BACKBONES = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2, 2048),
    "resnet101": (tvm.resnet101, tvm.ResNet101_Weights.IMAGENET1K_V2, 2048),
}


class ResNetBackbone(nn.Module):
    def __init__(self, name: str = "resnet50", pretrained: bool = True, freeze_stages: int = 6):
        super().__init__()
        if name not in _BACKBONES:
            raise ValueError(f"Unknown backbone {name!r}, choose from {list(_BACKBONES)}")
        ctor, weights, out_channels = _BACKBONES[name]
        net = ctor(weights=weights if pretrained else None)

        # keep everything up to (and excluding) avgpool/fc so we retain spatial resolution
        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
        )
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_channels = out_channels

        self._freeze(freeze_stages)

    def _freeze(self, n_stages: int) -> None:
        stages = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for stage in stages[:max(0, n_stages)]:
            for p in stage.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns spatial feature map of shape (B, C, H, W)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def output_shape(self, image_size: int) -> Tuple[int, int, int]:
        stride = 32
        side = max(1, image_size // stride)
        return self.out_channels, side, side
