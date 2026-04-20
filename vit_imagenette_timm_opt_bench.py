#!/usr/bin/env python3
"""
Imagenette ViT benchmark using timm models with optimizer A/B:
  - AdamW
  - Muon (timm's built-in implementation)

This script keeps a simple step-based loop similar to the existing local
benchmarks so optimizer comparisons are easy to run.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Some torchvision builds fail import unless torchvision::nms schema exists.
_TORCHVISION_FALLBACK_LIB = None


def import_torchvision_datasets_and_transforms():
    global _TORCHVISION_FALLBACK_LIB

    try:
        from torchvision import datasets, transforms
        from torchvision.transforms import InterpolationMode
        return datasets, transforms, InterpolationMode
    except RuntimeError as e:
        if "operator torchvision::nms does not exist" not in str(e):
            raise
        if _TORCHVISION_FALLBACK_LIB is None:
            _TORCHVISION_FALLBACK_LIB = torch.library.Library("torchvision", "DEF")
            _TORCHVISION_FALLBACK_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        for name in list(sys.modules):
            if name.startswith("torchvision"):
                del sys.modules[name]
        from torchvision import datasets, transforms
        from torchvision.transforms import InterpolationMode
        return datasets, transforms, InterpolationMode


datasets, transforms, InterpolationMode = import_torchvision_datasets_and_transforms()


IMAGENETTE_320_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGENETTE_ARCHIVE_NAME = "imagenette2-320.tgz"
IMAGENETTE_EXTRACTED_DIR = "imagenette2-320"
IMAGENETTE_NUM_CLASSES = 10


def _lazy_import_timm():
    try:
        import timm
        from timm.data import resolve_model_data_config
        from timm.optim import create_optimizer_v2
    except ImportError as e:
        raise ImportError(
            "timm is required for this script. Install with: pip install timm"
        ) from e
    return timm, resolve_model_data_config, create_optimizer_v2


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(amp_bf16: bool, device: torch.device):
    if amp_bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, amp_bf16: bool) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=(device.type == "cuda"))
        y = y.to(device, non_blocking=(device.type == "cuda"))
        with autocast_context(amp_bf16=amp_bf16, device=device):
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.detach())
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.numel())

    model.train(was_training)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def cycle(loader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = 100.0 * done / total_size
        mb_done = done / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\rDownloading {dst.name}: {pct:5.1f}%  {mb_done:7.1f}/{mb_total:7.1f} MB", end="")

    urllib.request.urlretrieve(url, dst, reporthook=_report)
    print()


def _extract_tgz(archive_path: Path, dst_dir: Path) -> None:
    marker_dir = dst_dir / IMAGENETTE_EXTRACTED_DIR
    if marker_dir.is_dir() and (marker_dir / "train").is_dir() and (marker_dir / "val").is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_root = dst_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_target = (dst_root / member.name).resolve()
            if member_target != dst_root and dst_root not in member_target.parents:
                raise RuntimeError(f"Refusing to extract '{member.name}' outside '{dst_root}'")
        tar.extractall(path=dst_root)


def resolve_imagenette_root(root: str) -> str:
    root_path = Path(root)
    direct = root_path
    nested = root_path / IMAGENETTE_EXTRACTED_DIR

    if (direct / "train").is_dir() and (direct / "val").is_dir():
        return str(direct)
    if (nested / "train").is_dir() and (nested / "val").is_dir():
        return str(nested)

    raise FileNotFoundError(
        f"Could not find Imagenette train/val folders under '{root}'. "
        f"Expected either '{root}/train' and '{root}/val' or "
        f"'{root}/{IMAGENETTE_EXTRACTED_DIR}/train' and '{root}/{IMAGENETTE_EXTRACTED_DIR}/val'."
    )


def maybe_download_imagenette(root: str, download: bool) -> str:
    try:
        return resolve_imagenette_root(root)
    except FileNotFoundError:
        if not download:
            raise

    root_path = Path(root)
    archive_path = root_path / IMAGENETTE_ARCHIVE_NAME
    print(f"Downloading Imagenette-320 to {archive_path} ...")
    _download_file(IMAGENETTE_320_URL, archive_path)
    print(f"Extracting {archive_path} ...")
    _extract_tgz(archive_path, root_path)
    return resolve_imagenette_root(root)


def maybe_subset_dataset(ds, max_items: Optional[int], seed: int):
    if max_items is None or max_items <= 0 or max_items >= len(ds):
        return ds
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(ds), generator=g)[:max_items].tolist()
    return torch.utils.data.Subset(ds, perm)


def _resolve_interpolation(name: str):
    name = (name or "bicubic").lower()
    if name in {"nearest"}:
        return InterpolationMode.NEAREST
    if name in {"bilinear", "linear"}:
        return InterpolationMode.BILINEAR
    return InterpolationMode.BICUBIC


def make_imagenette_loaders(
    root: str,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    image_size: int,
    download: bool,
    train_subset: Optional[int],
    val_subset: Optional[int],
    seed: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    crop_pct: float,
    interpolation_name: str,
    train_crop_min_scale: Optional[float] = None,
    random_erasing_prob: float = 0.0,
):
    ds_root = maybe_download_imagenette(root, download=download)
    train_dir = os.path.join(ds_root, "train")
    val_dir = os.path.join(ds_root, "val")

    interp = _resolve_interpolation(interpolation_name)
    train_tf_list = []
    if train_crop_min_scale is None:
        train_tf_list.append(
            transforms.RandomResizedCrop(image_size, interpolation=interp)
        )
    else:
        train_tf_list.append(
            transforms.RandomResizedCrop(
                image_size,
                scale=(train_crop_min_scale, 1.0),
                interpolation=interp,
            )
        )
    train_tf_list.extend([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if random_erasing_prob > 0.0:
        train_tf_list.append(transforms.RandomErasing(p=random_erasing_prob, value="random"))

    val_resize = int(round(image_size / max(crop_pct, 1e-6)))
    train_tf = transforms.Compose(train_tf_list)
    val_tf = transforms.Compose([
        transforms.Resize(val_resize, interpolation=interp),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    train_ds = maybe_subset_dataset(train_ds, train_subset, seed=seed)
    val_ds = maybe_subset_dataset(val_ds, val_subset, seed=seed + 1)

    train_drop_last = len(train_ds) >= batch_size
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train_drop_last,
        persistent_workers=(num_workers > 0),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    optimizer: str = "adamw"  # adamw | muon
    model: str = "vit_base_patch16_224"
    pretrained: bool = False
    data_root: str = "./data/imagenette"
    download_imagenette: bool = False
    train_subset: Optional[int] = None
    val_subset: Optional[int] = None
    steps: int = 2000
    eval_every: int = 100
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 8
    image_size: int = 224
    seed: int = 1
    amp_bf16: bool = True
    lr: float = 2e-3
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8
    warmup_steps: int = 200
    warmup_fraction: Optional[float] = None
    lr_schedule: str = "cosine"
    flat_lr_fraction: float = 0.0
    min_lr_scale: float = 0.1
    grad_clip: Optional[float] = 1.0
    label_smoothing: float = 0.1
    train_crop_min_scale: Optional[float] = None
    random_erasing_prob: float = 0.0
    filter_bias_and_bn: bool = True
    muon_momentum: float = 0.95
    muon_nesterov: bool = False
    muon_eps: float = 1e-7
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str = "match_rms_adamw"
    muon_fallback_lr_scale: float = 1.0
    muon_scale_eps: bool = False
    muon_verbose: bool = False
    log_json: Optional[str] = None


def validate_train_config(cfg: TrainConfig) -> None:
    if cfg.optimizer not in {"adamw", "muon"}:
        raise ValueError(f"optimizer must be one of adamw/muon, got {cfg.optimizer}")
    if cfg.eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {cfg.eval_every}")
    if cfg.steps <= 0:
        raise ValueError(f"steps must be positive, got {cfg.steps}")
    if cfg.image_size <= 0:
        raise ValueError(f"image_size must be positive, got {cfg.image_size}")
    if cfg.lr <= 0.0:
        raise ValueError(f"lr must be > 0, got {cfg.lr}")
    if cfg.lr_schedule not in {"cosine", "cosine_to_zero", "flat_cosine"}:
        raise ValueError(
            f"lr_schedule must be one of cosine/cosine_to_zero/flat_cosine, got {cfg.lr_schedule}"
        )
    if cfg.warmup_fraction is not None and not 0.0 <= cfg.warmup_fraction < 1.0:
        raise ValueError(f"warmup_fraction must be in [0, 1), got {cfg.warmup_fraction}")
    if not 0.0 <= cfg.flat_lr_fraction < 1.0:
        raise ValueError(f"flat_lr_fraction must be in [0, 1), got {cfg.flat_lr_fraction}")
    if not 0.0 <= cfg.min_lr_scale <= 1.0:
        raise ValueError(f"min_lr_scale must be in [0, 1], got {cfg.min_lr_scale}")
    if not 0.0 <= cfg.label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {cfg.label_smoothing}")
    if cfg.train_crop_min_scale is not None and not 0.0 < cfg.train_crop_min_scale <= 1.0:
        raise ValueError(f"train_crop_min_scale must be in (0, 1], got {cfg.train_crop_min_scale}")
    if not 0.0 <= cfg.random_erasing_prob <= 1.0:
        raise ValueError(f"random_erasing_prob must be in [0, 1], got {cfg.random_erasing_prob}")
    if not 0.0 <= cfg.muon_momentum < 1.0:
        raise ValueError(f"muon_momentum must be in [0, 1), got {cfg.muon_momentum}")
    if cfg.muon_adjust_lr_fn not in {"original", "match_rms_adamw", "rms_to_rms"}:
        raise ValueError(
            f"muon_adjust_lr_fn must be one of original/match_rms_adamw/rms_to_rms, got {cfg.muon_adjust_lr_fn}"
        )


def resolve_warmup_steps(total_steps: int, warmup_steps: int, warmup_fraction: Optional[float]) -> int:
    if warmup_fraction is not None:
        return int(round(total_steps * warmup_fraction))
    return warmup_steps


def _cosine_scale(progress: float, min_lr_scale: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_scale + (1.0 - min_lr_scale) * cosine


def lr_schedule_scale(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_scale: float,
    schedule: str,
    flat_lr_fraction: float = 0.0,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return step / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0

    if schedule == "cosine_to_zero":
        min_lr_scale = 0.0

    if schedule == "flat_cosine":
        post_warmup = max(total_steps - warmup_steps, 1)
        flat_steps = int(round(post_warmup * flat_lr_fraction))
        decay_start = min(total_steps, warmup_steps + flat_steps)
        if step <= decay_start:
            return 1.0
        t = (step - decay_start) / max(total_steps - decay_start, 1)
        return _cosine_scale(t, min_lr_scale)

    t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return _cosine_scale(t, min_lr_scale)


def set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def create_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    _, _, create_optimizer_v2 = _lazy_import_timm()
    if cfg.optimizer == "adamw":
        return create_optimizer_v2(
            model,
            opt="adamw",
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.adamw_eps,
            filter_bias_and_bn=cfg.filter_bias_and_bn,
        )
    return create_optimizer_v2(
        model,
        opt="muon",
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.muon_momentum,
        nesterov=cfg.muon_nesterov,
        betas=cfg.betas,
        eps=cfg.muon_eps,
        ns_steps=cfg.muon_ns_steps,
        adjust_lr_fn=cfg.muon_adjust_lr_fn,
        fallback_lr_scale=cfg.muon_fallback_lr_scale,
        scale_eps=cfg.muon_scale_eps,
        verbose=cfg.muon_verbose,
        filter_bias_and_bn=cfg.filter_bias_and_bn,
        fallback_no_weight_decay=True,
    )


def train(cfg: TrainConfig) -> list[dict]:
    validate_train_config(cfg)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timm, resolve_model_data_config, _ = _lazy_import_timm()

    model_kwargs = dict(num_classes=IMAGENETTE_NUM_CLASSES)
    if cfg.image_size > 0:
        model_kwargs["img_size"] = cfg.image_size
    model = timm.create_model(cfg.model, pretrained=cfg.pretrained, **model_kwargs).to(device)
    data_cfg = resolve_model_data_config(model)

    mean = tuple(data_cfg.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(data_cfg.get("std", (0.229, 0.224, 0.225)))
    crop_pct = float(data_cfg.get("crop_pct", 0.875))
    interpolation_name = str(data_cfg.get("interpolation", "bicubic"))

    train_loader, val_loader = make_imagenette_loaders(
        cfg.data_root,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
        download=cfg.download_imagenette,
        train_subset=cfg.train_subset,
        val_subset=cfg.val_subset,
        seed=cfg.seed,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        interpolation_name=interpolation_name,
        train_crop_min_scale=cfg.train_crop_min_scale,
        random_erasing_prob=cfg.random_erasing_prob,
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"Train loader has zero batches. Increase train_subset or lower batch_size (batch_size={cfg.batch_size})."
        )

    optimizer = create_optimizer(cfg, model)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    warmup_steps = resolve_warmup_steps(cfg.steps, cfg.warmup_steps, cfg.warmup_fraction)

    history: list[dict] = []
    wall_t0 = time.time()
    train_iter = cycle(train_loader)
    model.train()

    for step in range(1, cfg.steps + 1):
        lr_scale = lr_schedule_scale(
            step,
            cfg.steps,
            warmup_steps=warmup_steps,
            min_lr_scale=cfg.min_lr_scale,
            schedule=cfg.lr_schedule,
            flat_lr_fraction=cfg.flat_lr_fraction,
        )
        cur_lr = cfg.lr * lr_scale
        set_optimizer_lr(optimizer, cur_lr)

        x, y = next(train_iter)
        x = x.to(device, non_blocking=(device.type == "cuda"))
        y = y.to(device, non_blocking=(device.type == "cuda"))

        model.zero_grad(set_to_none=True)
        with autocast_context(amp_bf16=cfg.amp_bf16, device=device):
            logits = model(x)
            loss = criterion(logits, y)
        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        optimizer.step()

        if step == 1 or step % cfg.eval_every == 0 or step == cfg.steps:
            elapsed = time.time() - wall_t0
            batch_acc = accuracy_top1(logits.detach(), y)
            val_loss, val_acc = evaluate(model, val_loader, device, amp_bf16=cfg.amp_bf16)
            record = dict(
                step=step,
                time=elapsed,
                optimizer=cfg.optimizer,
                lr=float(cur_lr),
                lr_scale=float(lr_scale),
                batch_loss=float(loss.detach()),
                batch_acc=float(batch_acc),
                val_loss=float(val_loss),
                val_acc=float(val_acc),
            )
            history.append(record)
            print(
                f"[{cfg.optimizer:7s}] step {step:5d}/{cfg.steps}  "
                f"lr {cur_lr:.3e}  batch_loss {record['batch_loss']:.4f}  "
                f"val_acc {100.0 * val_acc:5.2f}%  time {elapsed/60.0:6.1f} min"
            )
            if cfg.log_json is not None:
                path = Path(cfg.log_json)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(history, indent=2))

    return history


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Imagenette timm ViT optimizer benchmark (AdamW vs Muon)")
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--data-root", type=str, default="./data/imagenette")
    parser.add_argument("--download-imagenette", action="store_true")
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-amp-bf16", action="store_true")

    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adamw-eps", type=float, default=1e-8)

    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--warmup-fraction", type=float, default=None)
    parser.add_argument("--lr-schedule", choices=["cosine", "cosine_to_zero", "flat_cosine"], default="cosine")
    parser.add_argument("--flat-lr-fraction", type=float, default=0.0)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--train-crop-min-scale", type=float, default=None)
    parser.add_argument("--random-erasing-prob", type=float, default=0.0)
    parser.add_argument("--log-json", type=str, default=None)

    parser.set_defaults(filter_bias_and_bn=True, muon_nesterov=False, muon_scale_eps=False, muon_verbose=False)
    parser.add_argument("--filter-bias-and-bn", action="store_true")
    parser.add_argument("--no-filter-bias-and-bn", dest="filter_bias_and_bn", action="store_false")

    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-nesterov", action="store_true")
    parser.add_argument("--no-muon-nesterov", dest="muon_nesterov", action="store_false")
    parser.add_argument("--muon-eps", type=float, default=1e-7)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument(
        "--muon-adjust-lr-fn",
        choices=["original", "match_rms_adamw", "rms_to_rms"],
        default="match_rms_adamw",
    )
    parser.add_argument("--muon-fallback-lr-scale", type=float, default=1.0)
    parser.add_argument("--muon-scale-eps", action="store_true")
    parser.add_argument("--muon-verbose", action="store_true")

    args = parser.parse_args()
    return TrainConfig(
        optimizer=args.optimizer,
        model=args.model,
        pretrained=args.pretrained,
        data_root=args.data_root,
        download_imagenette=args.download_imagenette,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        steps=args.steps,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=args.seed,
        amp_bf16=not args.no_amp_bf16,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        adamw_eps=args.adamw_eps,
        warmup_steps=args.warmup_steps,
        warmup_fraction=args.warmup_fraction,
        lr_schedule=args.lr_schedule,
        flat_lr_fraction=args.flat_lr_fraction,
        min_lr_scale=args.min_lr_scale,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
        train_crop_min_scale=args.train_crop_min_scale,
        random_erasing_prob=args.random_erasing_prob,
        filter_bias_and_bn=args.filter_bias_and_bn,
        muon_momentum=args.muon_momentum,
        muon_nesterov=args.muon_nesterov,
        muon_eps=args.muon_eps,
        muon_ns_steps=args.muon_ns_steps,
        muon_adjust_lr_fn=args.muon_adjust_lr_fn,
        muon_fallback_lr_scale=args.muon_fallback_lr_scale,
        muon_scale_eps=args.muon_scale_eps,
        muon_verbose=args.muon_verbose,
        log_json=args.log_json,
    )


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
