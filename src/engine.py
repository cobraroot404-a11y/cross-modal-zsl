"""Training and zero-shot evaluation loops."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import classification_report_dict, compute_confusion_matrix, harmonic_mean


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    class_attributes: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: Optional[float] = None,
    log_every: int = 20,
    epoch: int = 0,
) -> Dict[str, float]:
    model.train()
    class_attributes = class_attributes.to(device)

    running = {"loss": 0.0, "ce_loss": 0.0, "consistency_loss": 0.0}
    n_batches = 0

    pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    for images, labels, _global_ids in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        scores, _attn = model(images, class_attributes)
        losses = criterion(scores, labels)
        losses["loss"].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        n_batches += 1
        for k in running:
            running[k] += float(losses[k].detach())
        if n_batches % log_every == 0:
            pbar.set_postfix({k: v / n_batches for k, v in running.items()})

    return {k: v / max(1, n_batches) for k, v in running.items()}


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    class_attributes: torch.Tensor,
    class_ids: List[int],
    device: torch.device,
):
    """Returns (y_true_global, y_pred_global) using argmax cosine similarity
    over the given candidate class set (class_ids <-> rows of class_attributes)."""
    model.eval()
    class_attributes = class_attributes.to(device)

    y_true, y_pred = [], []
    for images, _local_labels, global_ids in loader:
        images = images.to(device, non_blocking=True)
        scores, _attn = model(images, class_attributes)
        pred_idx = scores.argmax(dim=-1).cpu().tolist()
        y_pred.extend(class_ids[i] for i in pred_idx)
        y_true.extend(global_ids.tolist())
    return y_true, y_pred


def evaluate_zero_shot(
    model: nn.Module,
    device: torch.device,
    seen_loader: Optional[DataLoader],
    unseen_loader: DataLoader,
    seen_class_ids: List[int],
    unseen_class_ids: List[int],
    attribute_tensor_fn,
) -> Dict:
    """Standard ZSL / GZSL evaluation protocol.

    ZSL setting  : classify unseen-class test images among unseen classes only.
    GZSL setting : classify seen- and unseen-class test images among the
                   union of both class sets, then report seen accuracy,
                   unseen accuracy and their harmonic mean H.
    """
    results: Dict = {}

    # --- ZSL: unseen images, unseen candidate classes only ---
    unseen_attrs = attribute_tensor_fn(unseen_class_ids)
    y_true_u, y_pred_u = run_inference(model, unseen_loader, unseen_attrs, unseen_class_ids, device)
    results["zsl"] = classification_report_dict(y_true_u, y_pred_u)
    results["zsl"]["confusion_matrix"] = compute_confusion_matrix(
        y_true_u, y_pred_u, labels=unseen_class_ids
    ).tolist()

    if seen_loader is not None:
        # --- GZSL: candidates = seen + unseen classes ---
        all_class_ids = seen_class_ids + unseen_class_ids
        all_attrs = attribute_tensor_fn(all_class_ids)

        y_true_s, y_pred_s = run_inference(model, seen_loader, all_attrs, all_class_ids, device)
        y_true_u2, y_pred_u2 = run_inference(model, unseen_loader, all_attrs, all_class_ids, device)

        acc_s = classification_report_dict(y_true_s, y_pred_s)["accuracy"]
        acc_u = classification_report_dict(y_true_u2, y_pred_u2)["accuracy"]
        results["gzsl"] = {
            "seen_accuracy": acc_s,
            "unseen_accuracy": acc_u,
            "harmonic_mean": harmonic_mean(acc_s, acc_u),
        }

    return results
