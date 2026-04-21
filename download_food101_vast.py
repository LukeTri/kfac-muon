#!/usr/bin/env python3
"""
Prepare Food-101 dataset in ImageFolder layout for timm train.py.

This script downloads Food-101 via torchvision and materializes:
  <out_root>/train/<class_name>/*.jpg
  <out_root>/val/<class_name>/*.jpg
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and prepare Food-101 for timm train.py.")
    p.add_argument("--source-root", type=str, default="/workspace/data/torchvision")
    p.add_argument("--out-root", type=str, default="/workspace/data/food101")
    p.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report-every", type=int, default=5000)
    return p.parse_args()


def _iter_rel_paths(list_file: Path) -> Iterable[str]:
    for line in list_file.read_text().splitlines():
        line = line.strip()
        if line:
            yield line


def _resolve_src_image(images_dir: Path, rel_path: str) -> Path:
    p = images_dir / rel_path
    if p.exists():
        return p
    jpg = images_dir / f"{rel_path}.jpg"
    if jpg.exists():
        return jpg
    raise FileNotFoundError(f"Missing source image for '{rel_path}' under {images_dir}")


def _materialize_file(src: Path, dst: Path, link_mode: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "symlink":
        os.symlink(src, dst)
    elif link_mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def _materialize_split(
    *,
    images_dir: Path,
    list_file: Path,
    out_dir: Path,
    link_mode: str,
    overwrite: bool,
    report_every: int,
) -> int:
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for rel in _iter_rel_paths(list_file):
        src = _resolve_src_image(images_dir, rel)
        cls = rel.split("/", 1)[0]
        stem = Path(rel).name
        if src.suffix:
            file_name = f"{stem}{src.suffix}" if Path(stem).suffix == "" else stem
        else:
            file_name = f"{stem}.jpg"
        dst = out_dir / cls / file_name
        _materialize_file(src, dst, link_mode=link_mode)
        count += 1
        if count % report_every == 0:
            print(f"  wrote {count} -> {out_dir}")
    print(f"  wrote {count} -> {out_dir}")
    return count


def main() -> None:
    args = parse_args()

    try:
        from torchvision.datasets import Food101
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "torchvision is required for Food-101 download. "
            "Install with: pip install torchvision"
        ) from e

    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("[download] Ensuring Food-101 is present via torchvision...")
    Food101(root=str(source_root), split="train", download=True)
    Food101(root=str(source_root), split="test", download=True)

    base = source_root / "food-101"
    images_dir = base / "images"
    meta_dir = base / "meta"
    train_list = meta_dir / "train.txt"
    test_list = meta_dir / "test.txt"

    if not images_dir.exists() or not train_list.exists() or not test_list.exists():
        raise FileNotFoundError(
            f"Food-101 files missing under {base}. Expected images/ and meta/train.txt + meta/test.txt."
        )

    print(f"[prep] train split ({args.link_mode})")
    train_count = _materialize_split(
        images_dir=images_dir,
        list_file=train_list,
        out_dir=out_root / "train",
        link_mode=args.link_mode,
        overwrite=args.overwrite,
        report_every=max(1, args.report_every),
    )
    print(f"[prep] val split from test ({args.link_mode})")
    val_count = _materialize_split(
        images_dir=images_dir,
        list_file=test_list,
        out_dir=out_root / "val",
        link_mode=args.link_mode,
        overwrite=args.overwrite,
        report_every=max(1, args.report_every),
    )

    train_classes = len([p for p in (out_root / "train").iterdir() if p.is_dir()])
    val_classes = len([p for p in (out_root / "val").iterdir() if p.is_dir()])

    print("[done] Food-101 prepared for ImageFolder")
    print(f"  root: {out_root}")
    print(f"  train images: {train_count}")
    print(f"  val images: {val_count}")
    print(f"  train classes: {train_classes}")
    print(f"  val classes: {val_classes}")
    print("  expected: train=75750, val=25250, classes=101")


if __name__ == "__main__":
    main()
