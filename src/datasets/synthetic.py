"""Synthetic drop-in replacement for CUBZeroShotDataset.

Generates procedural colour/shape images whose visual statistics correlate
with a random per-class attribute vector, so the cross-modal alignment
objective has real signal to learn. This lets the full pipeline (data
loading -> augmentation -> model -> loss -> zero-shot evaluation) be
smoke-tested in seconds without downloading CUB-200-2011.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .cub_zsl import ClassIndexer


def make_synthetic_world(
    num_classes: int = 20,
    num_attributes: int = 32,
    seed: int = 42,
) -> np.ndarray:
    """Return a (num_classes, num_attributes) attribute matrix in [0, 1]."""
    rng = np.random.RandomState(seed)
    return rng.rand(num_classes, num_attributes).astype(np.float32)


class SyntheticZSLDataset(Dataset):
    """Renders one procedural image per sample, colour/shape driven by class attributes."""

    def __init__(
        self,
        attributes: np.ndarray,
        class_ids: Sequence[int],
        indexer: ClassIndexer,
        samples_per_class: int = 12,
        image_size: int = 224,
        transform=None,
        seed: int = 0,
    ):
        self.attributes = attributes
        self.class_ids = list(class_ids)
        self.indexer = indexer
        self.image_size = image_size
        self.transform = transform
        self.samples_per_class = samples_per_class
        self._seed = seed

        self.index: List[Tuple[int, int]] = []
        for cid in self.class_ids:
            for k in range(samples_per_class):
                self.index.append((cid, k))

    def __len__(self) -> int:
        return len(self.index)

    def _render(self, class_id: int, sample_seed: int) -> Image.Image:
        attrs = self.attributes[class_id - 1]
        rng = np.random.RandomState(hash((class_id, sample_seed, self._seed)) % (2**31))

        r = int(50 + 205 * attrs[0])
        g = int(50 + 205 * attrs[1 % len(attrs)])
        b = int(50 + 205 * attrs[2 % len(attrs)])
        bg = tuple(int(max(0, min(255, 255 - c + rng.randint(-20, 20)))) for c in (r, g, b))

        img = Image.new("RGB", (self.image_size, self.image_size), color=bg)
        draw = ImageDraw.Draw(img)

        n_shapes = 3 + int(attrs[3 % len(attrs)] * 5)
        for i in range(n_shapes):
            cx = rng.randint(0, self.image_size)
            cy = rng.randint(0, self.image_size)
            radius = int(10 + attrs[(4 + i) % len(attrs)] * 40)
            jitter = rng.randint(-15, 15, size=3)
            colour = tuple(int(max(0, min(255, c + j))) for c, j in zip((r, g, b), jitter))
            shape_choice = attrs[(5 + i) % len(attrs)]
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            if shape_choice > 0.5:
                draw.ellipse(bbox, fill=colour)
            else:
                draw.rectangle(bbox, fill=colour)
        return img

    def __getitem__(self, idx: int):
        class_id, sample_k = self.index[idx]
        image = self._render(class_id, sample_k)
        if self.transform is not None:
            image = self.transform(image)
        local_label = self.indexer.to_local(class_id)
        return image, local_label, class_id
