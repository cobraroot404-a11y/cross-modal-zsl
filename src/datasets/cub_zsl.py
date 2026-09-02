"""CUB-200-2011 dataset loader for attribute-based zero-shot classification.

Expects the official archive layout::

    <root>/
        images/<class_folder>/<image>.jpg
        images.txt
        image_class_labels.txt
        classes.txt
        train_test_split.txt
        attributes/attributes.txt
        attributes/class_attribute_labels_continuous.txt

Download with ``scripts/download_cub.py``.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_classes(root: str) -> Dict[int, str]:
    """Return {class_id (1-indexed): class_name} e.g. {1: '001.Black_footed_Albatross'}."""
    classes = {}
    for line in _read_lines(os.path.join(root, "classes.txt")):
        cid, name = line.split(maxsplit=1)
        classes[int(cid)] = name
    return classes


def load_class_attributes(root: str, num_attributes: int = 312) -> np.ndarray:
    """Return a (num_classes, num_attributes) float array normalized to [0, 1]."""
    path = os.path.join(root, "attributes", "class_attribute_labels_continuous.txt")
    matrix = np.loadtxt(path, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, num_attributes)
    return matrix / 100.0


def load_attribute_names(root: str) -> List[str]:
    path = os.path.join(root, "attributes", "attributes.txt")
    if not os.path.exists(path):
        return []
    names = []
    for line in _read_lines(path):
        parts = line.split(maxsplit=1)
        names.append(parts[1] if len(parts) > 1 else parts[0])
    return names


def load_image_index(root: str) -> List[Tuple[int, str]]:
    """Return list of (image_id, relative_path)."""
    out = []
    for line in _read_lines(os.path.join(root, "images.txt")):
        iid, path = line.split(maxsplit=1)
        out.append((int(iid), path))
    return out


def load_image_labels(root: str) -> Dict[int, int]:
    """Return {image_id: class_id}."""
    labels = {}
    for line in _read_lines(os.path.join(root, "image_class_labels.txt")):
        iid, cid = line.split()
        labels[int(iid)] = int(cid)
    return labels


def load_train_test_split(root: str) -> Dict[int, int]:
    """Return {image_id: is_training_image(0/1)} as defined by the official CUB split."""
    split = {}
    path = os.path.join(root, "train_test_split.txt")
    for line in _read_lines(path):
        iid, is_train = line.split()
        split[int(iid)] = int(is_train)
    return split


def resolve_seen_unseen_split(
    root: str,
    classes: Dict[int, str],
    split_dir: Optional[str] = None,
    unseen_ratio: float = 0.25,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Return (seen_class_ids, unseen_class_ids).

    If ``split_dir`` contains ``trainvalclasses.txt`` / ``testclasses.txt``
    (one class folder name per line, matching ``classes.txt``), those are
    used -- this lets you plug in a literature-standard proposed split.
    Otherwise a deterministic stratified random split is generated so the
    experiment is reproducible without any extra download.
    """
    name_to_id = {name: cid for cid, name in classes.items()}

    if split_dir is not None:
        trainval_path = os.path.join(split_dir, "trainvalclasses.txt")
        test_path = os.path.join(split_dir, "testclasses.txt")
        if os.path.exists(trainval_path) and os.path.exists(test_path):
            seen = [name_to_id[n] for n in _read_lines(trainval_path) if n in name_to_id]
            unseen = [name_to_id[n] for n in _read_lines(test_path) if n in name_to_id]
            return sorted(seen), sorted(unseen)

    all_ids = sorted(classes.keys())
    rng = random.Random(seed)
    shuffled = all_ids[:]
    rng.shuffle(shuffled)
    n_unseen = max(1, int(round(len(all_ids) * unseen_ratio)))
    unseen = sorted(shuffled[:n_unseen])
    seen = sorted(shuffled[n_unseen:])
    return seen, unseen


class ClassIndexer:
    """Maps a fixed, ordered set of global class_ids to contiguous 0..N-1 labels."""

    def __init__(self, class_ids: Sequence[int]):
        self.class_ids = list(class_ids)
        self._to_local = {cid: i for i, cid in enumerate(self.class_ids)}

    def __len__(self) -> int:
        return len(self.class_ids)

    def to_local(self, class_id: int) -> int:
        return self._to_local[class_id]

    def to_global(self, local_idx: int) -> int:
        return self.class_ids[local_idx]


@dataclass
class CUBMeta:
    root: str
    classes: Dict[int, str]
    attributes: np.ndarray  # (200, A), 1-indexed class_id -> row (class_id - 1)
    attribute_names: List[str]
    seen_ids: List[int]
    unseen_ids: List[int]

    @classmethod
    def build(
        cls,
        root: str,
        split_dir: Optional[str] = None,
        unseen_ratio: float = 0.25,
        seed: int = 42,
    ) -> "CUBMeta":
        classes = load_classes(root)
        attributes = load_class_attributes(root)
        attribute_names = load_attribute_names(root)
        seen_ids, unseen_ids = resolve_seen_unseen_split(
            root, classes, split_dir=split_dir, unseen_ratio=unseen_ratio, seed=seed
        )
        return cls(root, classes, attributes, attribute_names, seen_ids, unseen_ids)

    def attribute_vector(self, class_id: int) -> np.ndarray:
        return self.attributes[class_id - 1]

    def attribute_tensor(self, class_ids: Sequence[int]) -> torch.Tensor:
        return torch.from_numpy(np.stack([self.attribute_vector(c) for c in class_ids]))


class CUBZeroShotDataset(Dataset):
    """One image-classification split of CUB restricted to a given set of classes.

    mode:
        'train'       -> seen classes, official training images only
        'test_seen'   -> seen classes, official test images only (for GZSL)
        'test_unseen' -> unseen classes, all images (never trained on)
    """

    def __init__(
        self,
        meta: CUBMeta,
        indexer: ClassIndexer,
        mode: str,
        transform=None,
    ):
        assert mode in ("train", "test_seen", "test_unseen")
        self.meta = meta
        self.indexer = indexer
        self.mode = mode
        self.transform = transform

        image_index = load_image_index(meta.root)
        image_labels = load_image_labels(meta.root)
        split_flags = load_train_test_split(meta.root)

        wanted_classes = set(indexer.class_ids)
        want_train_flag: Optional[int]
        if mode == "train":
            want_train_flag = 1
        elif mode == "test_seen":
            want_train_flag = 0
        else:  # test_unseen: use every image, class was never trained on
            want_train_flag = None

        self.samples: List[Tuple[str, int]] = []
        for image_id, rel_path in image_index:
            class_id = image_labels[image_id]
            if class_id not in wanted_classes:
                continue
            if want_train_flag is not None and split_flags.get(image_id) != want_train_flag:
                continue
            self.samples.append((rel_path, class_id))

        if not self.samples:
            raise RuntimeError(
                f"No samples found for mode={mode!r}. Check dataset root and class split."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path, class_id = self.samples[idx]
        img_path = os.path.join(self.meta.root, "images", rel_path)
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        local_label = self.indexer.to_local(class_id)
        return image, local_label, class_id
