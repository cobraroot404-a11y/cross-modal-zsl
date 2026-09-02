#!/usr/bin/env python
"""Train the cross-modal alignment zero-shot classifier.

Examples
--------
Smoke test (no dataset download needed, runs in seconds on CPU):
    python train.py --synthetic --epochs 2

Real training on CUB-200-2011 (see scripts/download_cub.py to fetch it):
    python train.py --data-root data/CUB_200_2011 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import yaml

from src.build import build_bundle
from src.engine import evaluate_zero_shot, train_one_epoch
from src.losses import CrossModalAlignmentLoss
from src.models.cross_modal_net import CrossModalZSLNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data-root", default=None, help="Override data.root from config")
    p.add_argument("--split-dir", default=None, help="Override data.split_dir from config")
    p.add_argument("--synthetic", action="store_true", help="Use the offline synthetic dataset")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None, help="Checkpoint path override")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.split_dir:
        cfg["data"]["split_dir"] = args.split_dir
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size

    torch.manual_seed(cfg["data"]["seed"])
    device = torch.device(args.device)

    bundle = build_bundle(
        synthetic=args.synthetic,
        data_root=cfg["data"]["root"],
        split_dir=cfg["data"]["split_dir"],
        unseen_ratio=cfg["data"]["unseen_ratio"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["train"]["batch_size"],
        eval_batch_size=cfg["eval"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["data"]["seed"],
    )

    print(f"seen classes:   {len(bundle.seen_class_ids)}")
    print(f"unseen classes: {len(bundle.unseen_class_ids)}")
    print(f"train images:   {len(bundle.train_loader.dataset)}")

    model = CrossModalZSLNet(
        attribute_dim=bundle.attribute_dim,
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        freeze_backbone_stages=cfg["model"]["freeze_backbone_stages"],
        embed_dim=cfg["model"]["embed_dim"],
        attribute_hidden_dim=cfg["model"]["attribute_hidden_dim"],
    ).to(device)

    criterion = CrossModalAlignmentLoss(
        temperature=cfg["train"]["temperature"],
        consistency_weight=cfg["train"]["consistency_weight"],
        label_smoothing=cfg["train"]["label_smoothing"],
    )

    backbone_params = list(model.backbone.parameters())
    head_params = list(model.attention.parameters()) + list(model.semantic_encoder.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in backbone_params if p.requires_grad], "lr": cfg["train"]["backbone_lr"]},
            {"params": head_params, "lr": cfg["train"]["lr"]},
        ],
        weight_decay=cfg["train"]["weight_decay"],
    )

    seen_attrs = bundle.attribute_tensor_fn(bundle.seen_class_ids)

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    ckpt_path = args.output or os.path.join(cfg["train"]["checkpoint_dir"], "cross_modal_zsl.pt")

    history = []
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        stats = train_one_epoch(
            model, bundle.train_loader, seen_attrs, criterion, optimizer, device,
            grad_clip=cfg["train"]["grad_clip"], log_every=cfg["train"]["log_every"], epoch=epoch,
        )
        print(f"[epoch {epoch}] " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()))
        history.append(stats)

    results = evaluate_zero_shot(
        model, device,
        seen_loader=bundle.test_seen_loader,
        unseen_loader=bundle.test_unseen_loader,
        seen_class_ids=bundle.seen_class_ids,
        unseen_class_ids=bundle.unseen_class_ids,
        attribute_tensor_fn=bundle.attribute_tensor_fn,
    )
    print(json.dumps({k: v for k, v in results.items() if k != "zsl"}, indent=2))
    print("ZSL metrics:", {k: v for k, v in results["zsl"].items() if k != "confusion_matrix"})

    torch.save({
        "model_state": model.state_dict(),
        "config": cfg,
        "seen_class_ids": bundle.seen_class_ids,
        "unseen_class_ids": bundle.unseen_class_ids,
        "attribute_dim": bundle.attribute_dim,
    }, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/train_history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "final_eval": results}, f, indent=2)


if __name__ == "__main__":
    main()
