#!/usr/bin/env python3
"""
Prepare full ImageNet-1K for timm train.py in ImageFolder layout.

This script intentionally does not download ImageNet-1K. The dataset is
license-gated, so the usual VAST flow is:

  1. Put an existing ImageFolder tree or official ILSVRC/Kaggle archives in
     SOURCE_ROOT, usually persistent storage.
  2. Materialize OUT_ROOT as either symlinks or extracted files.
  3. Train with timm using --data-dir OUT_ROOT --dataset image_folder.

Supported source layouts:

  A) Already extracted ImageFolder:
       SOURCE_ROOT/train/<wnid>/*.JPEG
       SOURCE_ROOT/val/<wnid>/*.JPEG

  B) Official-style archives:
       SOURCE_ROOT/ILSVRC2012_img_train.tar
       SOURCE_ROOT/ILSVRC2012_img_val.tar
       plus one of:
         - SOURCE_ROOT/LOC_val_solution.csv
         - SOURCE_ROOT/val_wnids.txt
         - --val-wnids-file PATH

Output layout:

       OUT_ROOT/train/<wnid>/*.JPEG
       OUT_ROOT/val/<wnid>/*.JPEG
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG"}
DEFAULT_TRAIN_ARCHIVE = "ILSVRC2012_img_train.tar"
DEFAULT_VAL_ARCHIVE = "ILSVRC2012_img_val.tar"


@dataclass
class SplitReport:
    classes: int
    images: int


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {e.lower() for e in IMAGE_EXTS}


def _count_split(split_dir: Path) -> SplitReport:
    if not split_dir.exists():
        return SplitReport(classes=0, images=0)
    class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
    images = 0
    for class_dir in class_dirs:
        images += sum(1 for p in class_dir.rglob("*") if p.is_file() and _is_image(p))
    return SplitReport(classes=len(class_dirs), images=images)


def _split_has_images(split_dir: Path) -> bool:
    report = _count_split(split_dir)
    return report.classes > 0 and report.images > 0


def _validate_root(root: Path, expected_classes: int, strict: bool = False) -> bool:
    train = _count_split(root / "train")
    val = _count_split(root / "val")
    print(f"[check] root={root}")
    print(f"  train: classes={train.classes} images={train.images}")
    print(f"  val:   classes={val.classes} images={val.images}")
    ok = train.classes > 0 and train.images > 0 and val.classes > 0 and val.images > 0
    if strict:
        ok = ok and train.classes == expected_classes and val.classes == expected_classes
    if strict and not ok:
        print(f"[warn] strict validation expected {expected_classes} classes in both train and val")
    return ok


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _materialize_existing_split(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if _split_has_images(dst) and not overwrite:
            print(f"[skip] {dst} already contains images")
            return
        if overwrite:
            _remove_existing(dst)
        else:
            raise FileExistsError(f"{dst} exists but does not look complete; use --overwrite")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=True)
        print(f"[link] {dst} -> {src.resolve()}")
    elif mode == "copy":
        print(f"[copy] {src} -> {dst}")
        shutil.copytree(src, dst)
    else:
        raise ValueError(f"Unsupported mode={mode!r}")


def _safe_member_name(member: tarfile.TarInfo) -> str:
    name = Path(member.name).name
    if not name or name in {".", ".."}:
        raise ValueError(f"Unsafe tar member name: {member.name!r}")
    return name


def _copy_tar_member_file(src: tarfile.TarFile, member: tarfile.TarInfo, dst: Path) -> None:
    extracted = src.extractfile(member)
    if extracted is None:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with extracted, dst.open("wb") as out:
        shutil.copyfileobj(extracted, out, length=1024 * 1024)


def _extract_train_archive(train_archive: Path, out_train: Path, overwrite: bool, report_every: int) -> None:
    if _split_has_images(out_train) and not overwrite:
        print(f"[skip] {out_train} already contains images")
        return
    if overwrite and out_train.exists():
        shutil.rmtree(out_train)
    out_train.mkdir(parents=True, exist_ok=True)

    print(f"[extract] train archive: {train_archive}")
    with tarfile.open(train_archive, "r") as outer:
        class_members = [m for m in outer.getmembers() if m.isfile() and m.name.endswith(".tar")]
        if not class_members:
            raise RuntimeError(
                f"No per-class .tar files found in {train_archive}. "
                "Expected official ILSVRC2012_img_train.tar layout."
            )
        for idx, class_member in enumerate(class_members, start=1):
            wnid = Path(class_member.name).stem
            class_dir = out_train / wnid
            if _split_has_images(class_dir) and not overwrite:
                continue
            class_dir.mkdir(parents=True, exist_ok=True)
            fileobj = outer.extractfile(class_member)
            if fileobj is None:
                continue
            data = fileobj.read()
            with tarfile.open(fileobj=io.BytesIO(data), mode="r") as class_tar:
                for image_member in class_tar.getmembers():
                    if not image_member.isfile():
                        continue
                    name = _safe_member_name(image_member)
                    _copy_tar_member_file(class_tar, image_member, class_dir / name)
            if idx % report_every == 0 or idx == len(class_members):
                print(f"  extracted {idx}/{len(class_members)} train class archives")


def _mapping_from_solution_csv(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "ImageId" not in reader.fieldnames or "PredictionString" not in reader.fieldnames:
            raise ValueError(f"{path} does not look like LOC_val_solution.csv")
        for row in reader:
            image_id = row["ImageId"].strip()
            pred = row["PredictionString"].strip().split()
            if not image_id or not pred:
                continue
            wnid = pred[0]
            mapping[f"{image_id}.JPEG"] = wnid
            mapping[image_id] = wnid
    return mapping


def _mapping_from_wnids_txt(path: Path, val_filenames: list[str]) -> dict[str, str]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    mapping: dict[str, str] = {}

    # Format 1: filename wnid
    pair_lines = [ln.split() for ln in lines]
    if pair_lines and all(len(parts) >= 2 for parts in pair_lines):
        for parts in pair_lines:
            mapping[Path(parts[0]).name] = parts[1]
        return mapping

    # Format 2: one WNID per line in validation filename order.
    if len(lines) != len(val_filenames):
        raise ValueError(
            f"{path} has {len(lines)} WNIDs but val archive has {len(val_filenames)} files. "
            "Expected either 'filename wnid' pairs or one WNID per validation image."
        )
    for filename, wnid in zip(val_filenames, lines):
        mapping[filename] = wnid
    return mapping


def _find_val_mapping_file(source_root: Path, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    for name in (
        "LOC_val_solution.csv",
        "val_wnids.txt",
        "imagenet_val_wnids.txt",
        "ILSVRC2012_val_wnids.txt",
    ):
        candidate = source_root / name
        if candidate.exists():
            return candidate
    return None


def _load_val_mapping(mapping_file: Path, val_filenames: list[str]) -> dict[str, str]:
    if mapping_file.name.endswith(".csv"):
        mapping = _mapping_from_solution_csv(mapping_file)
    else:
        mapping = _mapping_from_wnids_txt(mapping_file, val_filenames)
    if not mapping:
        raise RuntimeError(f"No validation mapping entries loaded from {mapping_file}")
    return mapping


def _extract_val_archive(
    val_archive: Path,
    out_val: Path,
    mapping_file: Path,
    overwrite: bool,
    report_every: int,
) -> None:
    if _split_has_images(out_val) and not overwrite:
        print(f"[skip] {out_val} already contains images")
        return
    if overwrite and out_val.exists():
        shutil.rmtree(out_val)
    out_val.mkdir(parents=True, exist_ok=True)

    print(f"[scan] val archive: {val_archive}")
    with tarfile.open(val_archive, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
    members = sorted(members, key=lambda m: Path(m.name).name)
    filenames = [Path(m.name).name for m in members]
    mapping = _load_val_mapping(mapping_file, filenames)
    print(f"[map] validation labels: {mapping_file}")

    missing = [name for name in filenames if name not in mapping and Path(name).stem not in mapping]
    if missing:
        raise RuntimeError(
            f"Validation mapping missing {len(missing)} filenames; first missing={missing[0]!r}. "
            "Provide --val-wnids-file with 'filename wnid' pairs or one WNID per sorted val image."
        )

    print(f"[extract] val archive: {val_archive}")
    with tarfile.open(val_archive, "r") as tar:
        by_name = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        for idx, filename in enumerate(filenames, start=1):
            wnid = mapping.get(filename) or mapping[Path(filename).stem]
            member = by_name[filename]
            _copy_tar_member_file(tar, member, out_val / wnid / filename)
            if idx % report_every == 0 or idx == len(filenames):
                print(f"  extracted {idx}/{len(filenames)} val images")


def _find_archive(source_root: Path, name: str) -> Path | None:
    candidate = source_root / name
    if candidate.exists():
        return candidate
    matches = list(source_root.rglob(name))
    return matches[0] if matches else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare full ImageNet-1K as ImageFolder for timm.")
    p.add_argument("--source-root", type=str, default="/workspace/data/imagenet_source")
    p.add_argument("--out-root", type=str, default="/workspace/data/imagenet")
    p.add_argument("--mode", choices=["symlink", "copy"], default="symlink",
                   help="How to materialize an already-extracted train/val source tree.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--strict", action="store_true", help="Require exactly --expected-classes classes per split.")
    p.add_argument("--expected-classes", type=int, default=1000)
    p.add_argument("--train-archive", type=str, default=DEFAULT_TRAIN_ARCHIVE)
    p.add_argument("--val-archive", type=str, default=DEFAULT_VAL_ARCHIVE)
    p.add_argument("--val-wnids-file", type=str, default=None,
                   help="Validation labels. Supports LOC_val_solution.csv, 'filename wnid' pairs, or one WNID per val image.")
    p.add_argument("--report-every", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        ok = _validate_root(out_root, expected_classes=args.expected_classes, strict=args.strict)
        raise SystemExit(0 if ok else 1)

    if not source_root.exists():
        raise FileNotFoundError(
            f"source root does not exist: {source_root}\n"
            "Put full ImageNet train/val folders or official archives there, or set SOURCE_ROOT."
        )

    source_train = source_root / "train"
    source_val = source_root / "val"
    if _split_has_images(source_train) and _split_has_images(source_val):
        print("[source] found extracted ImageFolder train/val")
        _materialize_existing_split(source_train, out_root / "train", args.mode, args.overwrite)
        _materialize_existing_split(source_val, out_root / "val", args.mode, args.overwrite)
        ok = _validate_root(out_root, expected_classes=args.expected_classes, strict=args.strict)
        raise SystemExit(0 if ok else 1)

    train_archive = _find_archive(source_root, args.train_archive)
    val_archive = _find_archive(source_root, args.val_archive)
    if train_archive is None or val_archive is None:
        raise FileNotFoundError(
            "Could not find extracted train/val or official train/val archives.\n"
            f"  source_root={source_root}\n"
            f"  looked for {args.train_archive} and {args.val_archive}\n"
            "Expected either SOURCE_ROOT/train + SOURCE_ROOT/val or official ILSVRC2012 archives."
        )

    _extract_train_archive(train_archive, out_root / "train", args.overwrite, max(1, args.report_every))

    mapping_file = _find_val_mapping_file(source_root, args.val_wnids_file)
    if mapping_file is None:
        raise FileNotFoundError(
            "Found validation archive but no validation label mapping.\n"
            "Provide one of these:\n"
            "  --val-wnids-file /path/to/val_wnids.txt\n"
            "  SOURCE_ROOT/val_wnids.txt\n"
            "  SOURCE_ROOT/LOC_val_solution.csv\n"
            "val_wnids.txt may contain either 'filename wnid' pairs or one WNID per sorted validation image."
        )
    _extract_val_archive(val_archive, out_root / "val", mapping_file, args.overwrite, report_every=5000)

    ok = _validate_root(out_root, expected_classes=args.expected_classes, strict=args.strict)
    if ok:
        print("[done] ImageNet-1K prepared")
        print(f"  root: {out_root}")
        print("  use: python train.py --data-dir {out_root} --dataset image_folder --train-split train --val-split val ...".format(out_root=out_root))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
