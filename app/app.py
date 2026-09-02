#!/usr/bin/env python
"""Flask demo app: upload a bird photo, see the zero-shot prediction among
classes the model was never trained on, plus the attribute-attention
heatmap that drove the decision.

Run:
    python app/app.py --checkpoint checkpoints/cross_modal_zsl.pt --data-root data/CUB_200_2011
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import uuid

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.cub_zsl import CUBMeta  # noqa: E402
from src.datasets.transforms import build_eval_transform  # noqa: E402
from src.models.cross_modal_net import CrossModalZSLNet  # noqa: E402

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
STATE = {"model": None, "meta": None, "ckpt": None, "device": None, "transform": None, "error": None}


def load_state(checkpoint_path: str, data_root: str, device_name: str) -> None:
    try:
        device = torch.device(device_name)
        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg = ckpt["config"]
        meta = CUBMeta.build(
            data_root,
            split_dir=cfg["data"]["split_dir"],
            unseen_ratio=cfg["data"]["unseen_ratio"],
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
        model.eval()

        STATE.update({
            "model": model,
            "meta": meta,
            "ckpt": ckpt,
            "device": device,
            "transform": build_eval_transform(cfg["data"]["image_size"]),
            "temperature": cfg["train"]["temperature"],
            "error": None,
        })
    except Exception as e:  # pragma: no cover - surfaced in the UI instead
        STATE["error"] = str(e)


def run_prediction(image: Image.Image, candidate_mode: str, topk: int = 5):
    meta = STATE["meta"]
    ckpt = STATE["ckpt"]
    model = STATE["model"]
    device = STATE["device"]

    if candidate_mode == "seen":
        candidate_ids = ckpt["seen_class_ids"]
    elif candidate_mode == "all":
        candidate_ids = ckpt["seen_class_ids"] + ckpt["unseen_class_ids"]
    else:
        candidate_ids = ckpt["unseen_class_ids"]

    attrs = meta.attribute_tensor(candidate_ids).to(device)
    x = STATE["transform"](image).unsqueeze(0).to(device)

    with torch.no_grad():
        scores, attn_map = model(x, attrs)
        probs = F.softmax(scores / STATE["temperature"], dim=-1)[0].cpu().numpy()

    order = np.argsort(-probs)[:topk]
    predictions = []
    for idx in order:
        cid = candidate_ids[int(idx)]
        name = meta.classes[cid].split(".", 1)[-1].replace("_", " ")
        predictions.append({"name": name, "confidence": round(float(probs[idx]) * 100, 2)})

    best_idx = int(order[0])
    heat = attn_map[0, best_idx].cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_img = Image.fromarray(np.uint8(cm.jet(heat)[:, :, :3] * 255)).resize(image.size)
    overlay = Image.blend(image.convert("RGB"), heat_img, alpha=0.45)

    return predictions, overlay


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", error=STATE["error"], result=None)


@app.route("/predict", methods=["POST"])
def predict():
    if STATE["error"] or STATE["model"] is None:
        return redirect(url_for("index"))

    file = request.files.get("image")
    candidate_mode = request.form.get("candidates", "unseen")
    if not file or file.filename == "":
        return redirect(url_for("index"))

    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    uid = uuid.uuid4().hex[:8]
    input_name = f"{uid}_in.png"
    overlay_name = f"{uid}_att.png"
    image.save(os.path.join(UPLOAD_DIR, input_name))

    predictions, overlay = run_prediction(image, candidate_mode)
    overlay.save(os.path.join(UPLOAD_DIR, overlay_name))

    result = {
        "input_image": url_for("static", filename=f"uploads/{input_name}"),
        "overlay_image": url_for("static", filename=f"uploads/{overlay_name}"),
        "predictions": predictions,
        "candidates": candidate_mode,
    }
    return render_template("index.html", error=None, result=result)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/cross_modal_zsl.pt")
    p.add_argument("--data-root", default="data/CUB_200_2011")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        STATE["error"] = (
            f"No checkpoint found at '{args.checkpoint}'. Train one first, e.g.:\n"
            f"  python train.py --synthetic --epochs 5   (quick demo)\n"
            f"  python train.py --data-root data/CUB_200_2011   (real training)"
        )
    else:
        load_state(args.checkpoint, args.data_root, args.device)
    app.run(host=args.host, port=args.port, debug=args.debug)
