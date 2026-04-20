#!/usr/bin/env python3
"""
Download an ImageNet-100 subset from Hugging Face and materialize ImageFolder layout.

Default source dataset:
  clane9/imagenet-100

Output layout (timm train.py friendly):
  <out_root>/train/<class_name>/*.jpg
  <out_root>/val/<class_name>/*.jpg
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset


@dataclass
class SplitSpec:
    src: str
    dst: str


def _safe_class_name(name: str) -> str:
    # Keep class names filesystem-safe while retaining readability.
    return name.replace("/", "_").strip()


def _resolve_label_name(label_value: object, label_feature: object) -> str:
    if isinstance(label_value, int) and isinstance(label_feature, ClassLabel):
        return _safe_class_name(label_feature.names[label_value])
    return _safe_class_name(str(label_value))


def _as_pil_rgb(image_obj: object) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    raise TypeError(
        f"Unsupported image object type {type(image_obj)!r}. "
        "Expected PIL image in dataset column."
    )


def _iter_split_items(ds: Dataset, image_key: str, label_key: str) -> Iterable[tuple[int, Image.Image, str]]:
    label_feature = ds.features[label_key]
    for idx, ex in enumerate(ds):
        img = _as_pil_rgb(ex[image_key])
        class_name = _resolve_label_name(ex[label_key], label_feature)
        yield idx, img, class_name


def _materialize_split(
    ds: Dataset,
    out_split_dir: Path,
    image_key: str,
    label_key: str,
    overwrite_split: bool,
    report_every: int,
) -> None:
    if out_split_dir.exists() and any(out_split_dir.rglob("*.jpg")) and not overwrite_split:
        print(f"[skip] {out_split_dir} already has jpg files (use --overwrite-split to rebuild)")
        return

    if overwrite_split and out_split_dir.exists():
        shutil.rmtree(out_split_dir)

    out_split_dir.mkdir(parents=True, exist_ok=True)

    total = len(ds)
    for idx, img, class_name in _iter_split_items(ds, image_key=image_key, label_key=label_key):
        class_dir = out_split_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / f"{idx:07d}.jpg"
        img.save(out_path, format="JPEG", quality=95)
        if (idx + 1) % report_every == 0 or (idx + 1) == total:
            print(f"  wrote {idx + 1}/{total} -> {out_split_dir}")


def _pick_present_splits(ds_dict: DatasetDict, train_split: str, val_split: str) -> list[SplitSpec]:
    specs: list[SplitSpec] = []
    if train_split in ds_dict:
        specs.append(SplitSpec(src=train_split, dst="train"))
    if val_split in ds_dict:
        specs.append(SplitSpec(src=val_split, dst="val"))
    elif "val" in ds_dict:
        specs.append(SplitSpec(src="val", dst="val"))
    if not specs:
        raise ValueError(
            f"No expected splits found. Available splits: {list(ds_dict.keys())}. "
            f"Tried train='{train_split}', val='{val_split}', and 'val'."
        )
    return specs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and prepare ImageNet-100 as ImageFolder.")
    p.add_argument("--dataset-id", type=str, default="clane9/imagenet-100")
    p.add_argument("--out-root", type=str, default="/workspace/data/imagenet100")
    p.add_argument("--cache-dir", type=str, default="/workspace/.cache/huggingface")
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="validation")
    p.add_argument("--image-key", type=str, default="image")
    p.add_argument("--label-key", type=str, default="label")
    p.add_argument("--overwrite-split", action="store_true")
    p.add_argument("--report-every", type=int, default=2000)
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[load] dataset={args.dataset_id}")
    ds_dict = load_dataset(
        args.dataset_id,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    if not isinstance(ds_dict, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(ds_dict)!r}")

    for spec in _pick_present_splits(ds_dict, train_split=args.train_split, val_split=args.val_split):
        print(f"[prep] {spec.src} -> {out_root / spec.dst}")
        _materialize_split(
            ds_dict[spec.src],
            out_root / spec.dst,
            image_key=args.image_key,
            label_key=args.label_key,
            overwrite_split=args.overwrite_split,
            report_every=max(1, args.report_every),
        )

    train_classes = len([p for p in (out_root / "train").iterdir() if p.is_dir()]) if (out_root / "train").exists() else 0
    val_classes = len([p for p in (out_root / "val").iterdir() if p.is_dir()]) if (out_root / "val").exists() else 0
    print("[done] ImageNet-100 prepared")
    print(f"  root: {out_root}")
    print(f"  train classes: {train_classes}")
    print(f"  val classes: {val_classes}")
    print("  expected for ImageNet-100: around 100 classes")


if __name__ == "__main__":
    main()
