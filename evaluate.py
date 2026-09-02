#!/usr/bin/env python
"""Evaluate a trained checkpoint: ZSL + GZSL accuracy, precision/recall/F1,
confusion matrix (saved as a PNG heatmap).

Example:
    python evaluate.py --checkpoint checkpoints/cross_modal_zsl.pt --data-root data/CUB_200_2011
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.build import build_bundle
from src.engine import evaluate_zero_shot
from src.models.cross_modal_net import CrossModalZSLNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--split-dir", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    bundle = build_bundle(
        synthetic=args.synthetic,
        data_root=args.data_root or cfg["data"]["root"],
        split_dir=args.split_dir or cfg["data"]["split_dir"],
        unseen_ratio=cfg["data"]["unseen_ratio"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["train"]["batch_size"],
        eval_batch_size=cfg["eval"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["data"]["seed"],
    )

    model = CrossModalZSLNet(
        attribute_dim=ckpt["attribute_dim"],
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        freeze_backbone_stages=cfg["model"]["freeze_backbone_stages"],
        embed_dim=cfg["model"]["embed_dim"],
        attribute_hidden_dim=cfg["model"]["attribute_hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    results = evaluate_zero_shot(
        model, device,
        seen_loader=bundle.test_seen_loader,
        unseen_loader=bundle.test_unseen_loader,
        seen_class_ids=bundle.seen_class_ids,
        unseen_class_ids=bundle.unseen_class_ids,
        attribute_tensor_fn=bundle.attribute_tensor_fn,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print(json.dumps(results, indent=2)[:2000])

    cm = np.array(results["zsl"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_title("Zero-Shot Confusion Matrix (unseen classes)")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)
    print(f"saved confusion matrix plot to {args.output_dir}/confusion_matrix.png")

    with open(os.path.join(args.output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
