"""Wires together dataset + loaders for either real CUB data or the
offline synthetic dataset (used for --synthetic smoke tests / CI)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from torch.utils.data import DataLoader

from .datasets.cub_zsl import ClassIndexer, CUBMeta, CUBZeroShotDataset
from .datasets.synthetic import SyntheticZSLDataset, make_synthetic_world
from .datasets.transforms import build_eval_transform, build_train_transform


@dataclass
class Bundle:
    attribute_dim: int
    seen_class_ids: List[int]
    unseen_class_ids: List[int]
    attribute_tensor_fn: Callable[[List[int]], torch.Tensor]
    train_loader: DataLoader
    test_seen_loader: DataLoader
    test_unseen_loader: DataLoader


def build_bundle(
    synthetic: bool,
    data_root: str,
    split_dir: Optional[str],
    unseen_ratio: float,
    image_size: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
    synthetic_num_classes: int = 20,
    synthetic_attr_dim: int = 32,
    synthetic_samples_per_class: int = 16,
) -> Bundle:
    train_tf = build_train_transform(image_size)
    eval_tf = build_eval_transform(image_size)

    if synthetic:
        attrs = make_synthetic_world(synthetic_num_classes, synthetic_attr_dim, seed=seed)
        all_ids = list(range(1, synthetic_num_classes + 1))
        n_unseen = max(1, int(round(synthetic_num_classes * unseen_ratio)))
        unseen_ids = all_ids[:n_unseen]
        seen_ids = all_ids[n_unseen:]

        seen_indexer = ClassIndexer(seen_ids)
        unseen_indexer = ClassIndexer(unseen_ids)

        train_ds = SyntheticZSLDataset(
            attrs, seen_ids, seen_indexer, samples_per_class=synthetic_samples_per_class,
            image_size=image_size, transform=train_tf, seed=1,
        )
        test_seen_ds = SyntheticZSLDataset(
            attrs, seen_ids, seen_indexer, samples_per_class=max(2, synthetic_samples_per_class // 4),
            image_size=image_size, transform=eval_tf, seed=2,
        )
        test_unseen_ds = SyntheticZSLDataset(
            attrs, unseen_ids, unseen_indexer, samples_per_class=max(2, synthetic_samples_per_class // 4),
            image_size=image_size, transform=eval_tf, seed=3,
        )

        def attribute_tensor_fn(class_ids: List[int]) -> torch.Tensor:
            return torch.from_numpy(attrs[[c - 1 for c in class_ids]])

        attribute_dim = synthetic_attr_dim

    else:
        meta = CUBMeta.build(data_root, split_dir=split_dir, unseen_ratio=unseen_ratio, seed=seed)
        seen_ids, unseen_ids = meta.seen_ids, meta.unseen_ids
        seen_indexer = ClassIndexer(seen_ids)
        unseen_indexer = ClassIndexer(unseen_ids)

        train_ds = CUBZeroShotDataset(meta, seen_indexer, mode="train", transform=train_tf)
        test_seen_ds = CUBZeroShotDataset(meta, seen_indexer, mode="test_seen", transform=eval_tf)
        test_unseen_ds = CUBZeroShotDataset(meta, unseen_indexer, mode="test_unseen", transform=eval_tf)

        attribute_tensor_fn = meta.attribute_tensor
        attribute_dim = meta.attributes.shape[1]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=len(train_ds) > batch_size,
    )
    test_seen_loader = DataLoader(
        test_seen_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
    )
    test_unseen_loader = DataLoader(
        test_unseen_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
    )

    return Bundle(
        attribute_dim=attribute_dim,
        seen_class_ids=seen_ids,
        unseen_class_ids=unseen_ids,
        attribute_tensor_fn=attribute_tensor_fn,
        train_loader=train_loader,
        test_seen_loader=test_seen_loader,
        test_unseen_loader=test_unseen_loader,
    )
