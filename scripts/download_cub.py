#!/usr/bin/env python
"""Downloads and extracts the CUB-200-2011 dataset (images + attributes).

Usage:
    python scripts/download_cub.py --dest data

This fetches CUB_200_2011.tgz (~1.1GB) from Caltech Vision and extracts it to
<dest>/CUB_200_2011/, which is the layout expected by src/datasets/cub_zsl.py.
If the automatic download fails (mirrors occasionally rate-limit or move),
download the archive manually from the Caltech Vision CUB-200-2011 page and
extract it so that <dest>/CUB_200_2011/images/ exists.
"""
from __future__ import annotations

import argparse
import os
import tarfile

import requests
from tqdm import tqdm

CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"


def download(url: str, dest_path: str) -> None:
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists, skipping download")
        return
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="downloading") as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))


def extract(archive_path: str, dest_dir: str) -> None:
    print(f"extracting {archive_path} -> {dest_dir}")
    with tarfile.open(archive_path) as tar:
        tar.extractall(dest_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", default="data", help="Directory to place CUB_200_2011/ into")
    p.add_argument("--url", default=CUB_URL)
    p.add_argument("--keep-archive", action="store_true")
    args = p.parse_args()

    archive_path = os.path.join(args.dest, "CUB_200_2011.tgz")
    try:
        download(args.url, archive_path)
    except Exception as e:  # pragma: no cover - depends on network availability
        print(f"Automatic download failed ({e}).")
        print("Please download the archive manually from the official Caltech Vision")
        print("CUB-200-2011 page and extract it so that <dest>/CUB_200_2011/images/ exists.")
        return

    extract(archive_path, args.dest)
    if not args.keep_archive:
        os.remove(archive_path)

    expected = os.path.join(args.dest, "CUB_200_2011", "images")
    if os.path.isdir(expected):
        print(f"done. dataset ready at {os.path.join(args.dest, 'CUB_200_2011')}")
    else:
        print("extraction finished but images/ was not found -- check the archive layout.")


if __name__ == "__main__":
    main()
