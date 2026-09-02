#!/usr/bin/env python
"""Single-image inference: predicts the most likely class among a chosen
candidate set (by default, the unseen classes held out from training) and
saves an attention heatmap overlay so you can see what the model looked at.

Example:
    python predict.py --checkpoint checkpoints/cross_modal_zsl.pt \
        --data-root data/CUB_200_2011 --image path/to/bird.jpg --candidates unseen
"""
from __future__ import annotations

import argparse
import os

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.datasets.cub_zsl import CUBMeta
from src.datasets.transforms import build_eval_transform
from src.models.cross_modal_net import CrossModalZSLNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True, help="CUB root, used for class names + attributes")
    p.add_argument("--image", required=True)
    p.add_argument("--candidates", choices=["seen", "unseen", "all"], default="unseen")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    meta = CUBMeta.build(
        args.data_root,
        split_dir=cfg["data"]["split_dir"],
        unseen_ratio=cfg["data"]["unseen_ratio"],
        seed=cfg["data"]["seed"],
    )

    if args.candidates == "seen":
        candidate_ids = ckpt["seen_class_ids"]
    elif args.candidates == "unseen":
        candidate_ids = ckpt["unseen_class_ids"]
    else:
        candidate_ids = ckpt["seen_class_ids"] + ckpt["unseen_class_ids"]

    attrs = meta.attribute_tensor(candidate_ids).to(device)

    model = CrossModalZSLNet(
        attribute_dim=ckpt["attribute_dim"],
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        freeze_backbone_stages=cfg["model"]["freeze_backbone_stages"],
        embed_dim=cfg["model"]["embed_dim"],
        attribute_hidden_dim=cfg["model"]["attribute_hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tf = build_eval_transform(cfg["data"]["image_size"])
    image = Image.open(args.image).convert("RGB")
    x = tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        scores, attn_map = model(x, attrs)
        probs = F.softmax(scores / cfg["train"]["temperature"], dim=-1)[0].cpu().numpy()

    order = np.argsort(-probs)[: args.topk]
    print(f"Top-{args.topk} predictions among {args.candidates} classes:")
    for rank, idx in enumerate(order, start=1):
        cid = candidate_ids[idx]
        name = meta.classes[cid].split(".", 1)[-1].replace("_", " ")
        print(f"  {rank}. {name:35s} ({probs[idx]*100:5.2f}%)")

    best_idx = int(order[0])
    heat = attn_map[0, best_idx].cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_img = Image.fromarray(np.uint8(cm.jet(heat)[:, :, :3] * 255)).resize(image.size)
    overlay = Image.blend(image, heat_img, alpha=0.45)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "attention_overlay.png")
    overlay.save(out_path)
    print(f"saved attention overlay to {out_path}")


if __name__ == "__main__":
    main()
