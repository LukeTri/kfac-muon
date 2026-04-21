#!/usr/bin/env python3
"""
Prepare Places365 in ImageFolder layout for timm train.py.

This script uses torchvision Places365 and materializes:
  <out_root>/train/<class_name>/*.{jpg,png,...}
  <out_root>/val/<class_name>/*.{jpg,png,...}

Defaults target the "small" (256px) Places365 standard split.
"""

from __future__ import annotations

import argparse
import errno
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and prepare Places365 for timm train.py.")
    p.add_argument("--source-root", type=str, default="/workspace/data/places365_raw")
    p.add_argument("--out-root", type=str, default="/workspace/data/places365")
    p.add_argument("--train-split", type=str, default="train-standard")
    p.add_argument("--val-split", type=str, default="val")
    p.add_argument("--small", action="store_true", default=True, help="Use 256px Places365 data.")
    p.add_argument("--large", dest="small", action="store_false", help="Use large-resolution Places365 data.")
    p.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report-every", type=int, default=20000)
    return p.parse_args()


def _safe_class_name(cls: str) -> str:
    # Example class key can look like "/a/airfield".
    cls = cls.lstrip("/")
    return cls.replace("/", "__")


def _materialize_file(src: Path, dst: Path, link_mode: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "symlink":
        os.symlink(src, dst)
    elif link_mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError as e:
            if e.errno == errno.EXDEV:
                shutil.copy2(src, dst)
            else:
                raise
    else:
        shutil.copy2(src, dst)


def _build_imagefolder_split(
    *,
    ds,
    out_dir: Path,
    link_mode: str,
    overwrite: bool,
    report_every: int,
) -> int:
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for src_file, target in ds.imgs:
        if target is None:
            continue
        class_name = _safe_class_name(ds.classes[int(target)])
        src = Path(src_file)
        dst = out_dir / class_name / src.name
        _materialize_file(src, dst, link_mode=link_mode)
        count += 1
        if count % report_every == 0:
            print(f"  wrote {count} -> {out_dir}")
    print(f"  wrote {count} -> {out_dir}")
    return count


def _load_places365(*, root: str, split: str, small: bool, download: bool):
    from torchvision.datasets import Places365

    try:
        return Places365(root=root, split=split, small=small, download=download)
    except RuntimeError as e:
        # Some torchvision versions complain if already extracted while download=True.
        msg = str(e).lower()
        if download and "already extracted" in msg:
            return Places365(root=root, split=split, small=small, download=False)
        raise


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[download] Places365 train split: {args.train_split} (small={args.small})")
    train_ds = _load_places365(
        root=str(source_root),
        split=args.train_split,
        small=args.small,
        download=True,
    )
    print(f"[download] Places365 val split: {args.val_split} (small={args.small})")
    val_ds = _load_places365(
        root=str(source_root),
        split=args.val_split,
        small=args.small,
        download=True,
    )

    print(f"[prep] train -> {out_root / 'train'} ({args.link_mode})")
    train_count = _build_imagefolder_split(
        ds=train_ds,
        out_dir=out_root / "train",
        link_mode=args.link_mode,
        overwrite=args.overwrite,
        report_every=max(1, args.report_every),
    )
    print(f"[prep] val -> {out_root / 'val'} ({args.link_mode})")
    val_count = _build_imagefolder_split(
        ds=val_ds,
        out_dir=out_root / "val",
        link_mode=args.link_mode,
        overwrite=args.overwrite,
        report_every=max(1, args.report_every),
    )

    train_classes = len([p for p in (out_root / "train").iterdir() if p.is_dir()])
    val_classes = len([p for p in (out_root / "val").iterdir() if p.is_dir()])
    print("[done] Places365 prepared for ImageFolder")
    print(f"  root: {out_root}")
    print(f"  train images: {train_count}")
    print(f"  val images: {val_count}")
    print(f"  train classes: {train_classes}")
    print(f"  val classes: {val_classes}")
    print("  expected classes: 365")


if __name__ == "__main__":
    main()
