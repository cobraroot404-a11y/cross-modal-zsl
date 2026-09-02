#!/usr/bin/env python
"""Writes out the resolved seen/unseen class split as trainvalclasses.txt /
testclasses.txt, so the exact zero-shot split used for a run is documented
and reproducible (and can be reused later via --split-dir).

Usage:
    python scripts/export_split.py --data-root data/CUB_200_2011 --out splits/default
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.cub_zsl import CUBMeta  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--unseen-ratio", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="splits/default")
    args = p.parse_args()

    meta = CUBMeta.build(args.data_root, split_dir=None, unseen_ratio=args.unseen_ratio, seed=args.seed)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "trainvalclasses.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(meta.classes[c] for c in meta.seen_ids) + "\n")
    with open(os.path.join(args.out, "testclasses.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(meta.classes[c] for c in meta.unseen_ids) + "\n")

    print(f"{len(meta.seen_ids)} seen / {len(meta.unseen_ids)} unseen classes written to {args.out}/")
    print("Reuse this exact split later with --split-dir", args.out)


if __name__ == "__main__":
    main()
