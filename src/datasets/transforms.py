"""Image preprocessing & augmentation pipeline (Module 1 & 2 of the report).

Implements normalization + train-time augmentation (random flip, rotation,
scale/crop, translation, gaussian noise) and a deterministic eval-time
pipeline (resize + center-crop + normalize).
"""
import random

import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GaussianNoise:
    """Adds zero-mean Gaussian noise to a tensor image (regularization augmentation)."""

    def __init__(self, std: float = 0.03, p: float = 0.3):
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return tensor + noise


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),  # scale + crop
        transforms.RandomHorizontalFlip(p=0.5),                       # flip
        transforms.RandomVerticalFlip(p=0.1),                         # flip
        transforms.RandomRotation(degrees=25),                        # rotation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),     # translation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        GaussianNoise(std=0.03, p=0.3),                               # gaussian noise
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
