#!/usr/bin/env python3
"""
Prepare CIFAR-100 dataset in ImageFolder layout for timm train.py.

This script downloads CIFAR-100 via torchvision and materializes:
  <out_root>/train/<class_name>/*.png
  <out_root>/val/<class_name>/*.png
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and prepare CIFAR-100 for timm train.py.")
    p.add_argument("--source-root", type=str, default="/workspace/data/torchvision")
    p.add_argument("--out-root", type=str, default="/workspace/data/cifar100")
    p.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="copy",
        help="Accepted for CLI parity with other dataset scripts; CIFAR-100 is always materialized as PNG files.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report-every", type=int, default=5000)
    return p.parse_args()


def _materialize_split(
    *,
    ds,
    out_dir: Path,
    overwrite: bool,
    report_every: int,
) -> int:
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = list(getattr(ds, "classes", []))
    if not class_names:
        class_names = [str(i) for i in range(100)]

    per_class_count = defaultdict(int)
    count = 0

    for image, label in ds:
        cls_name = str(class_names[int(label)]).replace("/", "_")
        file_idx = per_class_count[int(label)]
        per_class_count[int(label)] += 1
        dst = out_dir / cls_name / f"{file_idx:05d}.png"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            image.save(dst, format="PNG")
        count += 1
        if count % report_every == 0:
            print(f"  wrote {count} -> {out_dir}")

    print(f"  wrote {count} -> {out_dir}")
    return count


def main() -> None:
    args = parse_args()

    try:
        from torchvision.datasets import CIFAR100
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "torchvision is required for CIFAR-100 download. "
            "Install with: pip install torchvision"
        ) from e

    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.link_mode != "copy":
        print(
            f"[note] link-mode={args.link_mode} requested, but CIFAR-100 examples are generated files; "
            "writing PNG files instead."
        )

    print("[download] Ensuring CIFAR-100 is present via torchvision...")
    train_ds = CIFAR100(root=str(source_root), train=True, download=True)
    test_ds = CIFAR100(root=str(source_root), train=False, download=True)

    print("[prep] train split (PNG materialization)")
    train_count = _materialize_split(
        ds=train_ds,
        out_dir=out_root / "train",
        overwrite=args.overwrite,
        report_every=max(1, args.report_every),
    )
    print("[prep] val split from test (PNG materialization)")
    val_count = _materialize_split(
        ds=test_ds,
        out_dir=out_root / "val",
        overwrite=args.overwrite,
        report_every=max(1, args.report_every),
    )

    train_classes = len([p for p in (out_root / "train").iterdir() if p.is_dir()])
    val_classes = len([p for p in (out_root / "val").iterdir() if p.is_dir()])

    print("[done] CIFAR-100 prepared for ImageFolder")
    print(f"  root: {out_root}")
    print(f"  train images: {train_count}")
    print(f"  val images: {val_count}")
    print(f"  train classes: {train_classes}")
    print(f"  val classes: {val_classes}")
    print("  expected: train=50000, val=10000, classes=100")


if __name__ == "__main__":
    main()
