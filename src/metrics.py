"""Evaluation metrics (report Module 5: Evaluation Metrics + Ch.6 results).

Zero-shot learning is conventionally scored with *per-class mean accuracy*
(a.k.a. mean class accuracy) rather than plain top-1 accuracy, because test
classes are rarely balanced. We report that, plus macro precision/recall/F1
and the confusion matrix, and the standard ZSL/GZSL harmonic mean.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def per_class_mean_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    accs = []
    for c in classes:
        mask = y_true == c
        accs.append(np.mean(y_pred[mask] == y_true[mask]))
    return float(np.mean(accs))


def harmonic_mean(seen_acc: float, unseen_acc: float) -> float:
    if seen_acc + unseen_acc == 0:
        return 0.0
    return 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc)


def classification_report_dict(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    return {
        "accuracy": per_class_mean_accuracy(y_true, y_pred),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], labels=None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)
