#!/usr/bin/env python3
"""
Stage-1 ViT benchmark on Imagenette for:
  - plain Muon
  - reduce-style KFAC-Muon

This is a small smoke-test version of the ImageNet/ViT benchmark.
It keeps only what is needed:
  * Imagenette data loading, with optional auto-download/extract
  * a ViT with explicit nn.Linear attention / MLP projections
  * Muon on the hidden affine weights
  * KFAC-Muon (reduce only) on the same hidden affine weights
  * AdamW on the non-Muon parameters

Dataset layout after download/extract is expected to be one of:
  root/train, root/val
or
  root/imagenette2-320/train, root/imagenette2-320/val
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import tarfile
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

# This script only uses torchvision datasets/transforms. Some environments ship
# a torchvision build whose Python package expects the torchvision::nms operator
# schema to exist even when the compiled ops are unavailable. We only define a
# minimal schema as a fallback after a failed import, which avoids duplicate
# registration in environments where torchvision imports cleanly.
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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENETTE_320_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGENETTE_ARCHIVE_NAME = "imagenette2-320.tgz"
IMAGENETTE_EXTRACTED_DIR = "imagenette2-320"
IMAGENETTE_NUM_CLASSES = 10


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def autocast_context(amp_bf16: bool, device: torch.device):
    if amp_bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


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
    train_crop_min_scale: float = 0.35,
    random_erasing_prob: float = 0.0,
):
    ds_root = maybe_download_imagenette(root, download=download)
    train_dir = os.path.join(ds_root, "train")
    val_dir = os.path.join(ds_root, "val")

    train_tf_list = [
        transforms.RandomResizedCrop(
            image_size,
            scale=(train_crop_min_scale, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if random_erasing_prob > 0.0:
        train_tf_list.append(transforms.RandomErasing(p=random_erasing_prob, value="random"))
    train_tf_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    train_tf = transforms.Compose(train_tf_list)
    val_tf = transforms.Compose([
        transforms.Resize(int(round(image_size / 0.875)), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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
# ViT model
# -----------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        separate_qkv: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.separate_qkv = bool(separate_qkv)

        if self.separate_qkv:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.qkv = None
        else:
            self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        if self.separate_qkv:
            q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        separate_qkv: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            separate_qkv=separate_qkv,
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        pool: str = "mean",
        separate_qkv: bool = True,
    ):
        super().__init__()
        if pool not in {"mean", "cls"}:
            raise ValueError(f"Unknown pool={pool}")
        self.pool = pool
        self.use_cls_token = (pool == "cls")

        self.patch_embed = PatchEmbed(image_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        seq_len = num_patches + (1 if self.use_cls_token else 0)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                separate_qkv=separate_qkv,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        def _init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.use_cls_token:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.use_cls_token:
            return x[:, 0]
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_vit(
    model_name: str = "small",
    *,
    image_size: int = 224,
    patch_size: int = 16,
    num_classes: int = IMAGENETTE_NUM_CLASSES,
    pool: str = "mean",
    separate_qkv: bool = True,
) -> VisionTransformer:
    if model_name == "tiny":
        return VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            pool=pool,
            separate_qkv=separate_qkv,
        )
    if model_name == "small":
        return VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            pool=pool,
            separate_qkv=separate_qkv,
        )
    raise ValueError(f"Unknown model_name={model_name}")


# -----------------------------------------------------------------------------
# Muon utilities
# -----------------------------------------------------------------------------

def _candidate_affine_modules(model: nn.Module) -> list[nn.Module]:
    mods: list[nn.Module] = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if getattr(m, "weight", None) is not None and m.weight.requires_grad:
                mods.append(m)
        elif isinstance(m, nn.Conv2d):
            if getattr(m, "weight", None) is not None and m.weight.requires_grad and m.groups == 1:
                mods.append(m)
    return mods


def split_muon_and_aux(
    model: nn.Module,
    *,
    exclude_first_last: bool = True,
) -> tuple[list[nn.Module], list[nn.Parameter], list[nn.Parameter]]:
    affine_modules = _candidate_affine_modules(model)
    if exclude_first_last and len(affine_modules) >= 2:
        muon_modules = affine_modules[1:-1]
    else:
        muon_modules = affine_modules

    muon_param_ids = {id(m.weight) for m in muon_modules}
    muon_params: list[nn.Parameter] = []
    aux_params: list[nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in muon_param_ids:
            muon_params.append(p)
        else:
            aux_params.append(p)
    return muon_modules, muon_params, aux_params


def _use_no_weight_decay(name: str, p: nn.Parameter) -> bool:
    if p.ndim <= 1:
        return True
    if name == "pos_embed" or name.endswith(".pos_embed"):
        return True
    if name == "cls_token" or name.endswith(".cls_token"):
        return True
    return False


def build_aux_adamw_param_groups(
    model: nn.Module,
    muon_params: list[nn.Parameter],
    weight_decay: float,
) -> list[dict]:
    muon_param_ids = {id(p) for p in muon_params}
    seen: set[int] = set()
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in muon_param_ids or id(p) in seen:
            continue
        seen.add(id(p))
        if _use_no_weight_decay(name, p):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups: list[dict] = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return param_groups


def muon_quintic_ns(
    X: torch.Tensor,
    ns_steps: int = 5,
    ns_coefficients=(3.4445, -4.775, 2.0315),
    eps: float = 1e-7,
) -> torch.Tensor:
    a, b, c = ns_coefficients
    orig_dtype = X.dtype

    Y = X.float()
    transposed = False
    if Y.shape[-2] > Y.shape[-1]:
        Y = Y.transpose(-2, -1)
        transposed = True

    Y = Y / (Y.square().sum(dim=(-2, -1), keepdim=True).sqrt() + eps)
    for _ in range(ns_steps):
        A = Y @ Y.transpose(-2, -1)
        B = b * A + c * (A @ A)
        Y = a * Y + B @ Y

    if transposed:
        Y = Y.transpose(-2, -1)
    return Y.to(orig_dtype)


def _adjust_muon_lr(lr: float, matrix_shape: tuple[int, int], mode: str = "match_rms_adamw") -> float:
    m, n = matrix_shape
    if mode == "original":
        return lr * math.sqrt(max(1.0, m / n))
    if mode == "match_rms_adamw":
        return lr * (0.2 * math.sqrt(max(m, n)))
    raise ValueError(f"Unknown Muon LR adjustment mode: {mode}")


class FlattenedMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        eps: float = 1e-7,
        ns_steps: int = 5,
        ns_coefficients=(3.4445, -4.7750, 2.0315),
        lr_adjustment: str = "match_rms_adamw",
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            eps=eps,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            lr_adjustment=lr_adjustment,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            ns_coefficients = group["ns_coefficients"]
            lr_adjustment = group["lr_adjustment"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse:
                    raise RuntimeError("FlattenedMuon does not support sparse gradients")
                if p.ndim < 2:
                    raise RuntimeError(f"FlattenedMuon expects ndim >= 2 params, got {tuple(p.shape)}")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g, memory_format=torch.preserve_format)
                buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update_2d = update if update.ndim == 2 else update.reshape(update.shape[0], -1)

                ortho_2d = muon_quintic_ns(
                    update_2d,
                    ns_steps=ns_steps,
                    ns_coefficients=ns_coefficients,
                    eps=eps,
                )
                ortho = ortho_2d if update.ndim == 2 else ortho_2d.reshape_as(update)
                adj_lr = _adjust_muon_lr(lr, tuple(update_2d.shape), mode=lr_adjustment)

                p.mul_(1 - lr * weight_decay)
                p.add_(ortho.to(dtype=p.dtype), alpha=-adj_lr)
        return loss


# -----------------------------------------------------------------------------
# KFAC-reduce for Muon-updated affine modules only
# -----------------------------------------------------------------------------

@dataclass
class KFACConfig:
    damping: float = 1e-3
    ema_decay: float = 0.95
    stats_update_every: int = 50
    factor_update_every: int = 50
    momentum: float = 0.95
    nesterov: bool = True
    muon_eps: float = 0.1
    muon_ns_steps: int = 5
    muon_ns_eps: float = 1e-7
    muon_ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315)
    lr_adjustment: str = "none"
    weight_decay: float = 0.0
    max_step_norm: Optional[float] = None


class KFACReduce:
    def __init__(self, modules: list[nn.Module], cfg: KFACConfig):
        self.modules = list(modules)
        self.cfg = cfg
        self._cache = {m: {} for m in self.modules}
        self._hooks = []
        self._step = 0
        self.stats: dict[nn.Module, dict[str, torch.Tensor]] = {}
        self.factors: dict[nn.Module, dict[str, torch.Tensor]] = {}
        self.eye_A: dict[nn.Module, torch.Tensor] = {}
        self.eye_G: dict[nn.Module, torch.Tensor] = {}
        self.state: dict[nn.Module, dict[str, torch.Tensor]] = {m: {} for m in self.modules}
        self._init_buffers()
        self._register_hooks()

    @staticmethod
    def _weight_matrix_shape(m: nn.Module) -> tuple[int, int]:
        if isinstance(m, nn.Linear):
            return m.out_features, m.in_features
        if isinstance(m, nn.Conv2d):
            if m.groups != 1:
                raise NotImplementedError("KFACReduce only supports Conv2d with groups=1")
            kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
            return m.out_channels, m.in_channels * kh * kw
        raise TypeError(f"Unsupported module type: {type(m)}")

    def _init_buffers(self) -> None:
        if not self.modules:
            return
        device = next(self.modules[0].parameters()).device
        dtype = torch.float32
        root = math.sqrt(max(self.cfg.damping, 0.0))
        for m in self.modules:
            out_dim, in_dim = self._weight_matrix_shape(m)
            A0 = torch.eye(in_dim, device=device, dtype=dtype)
            G0 = torch.eye(out_dim, device=device, dtype=dtype)
            IA = torch.eye(in_dim, device=device, dtype=dtype)
            IG = torch.eye(out_dim, device=device, dtype=dtype)
            LA = torch.linalg.cholesky(A0 + root * IA)
            LG = torch.linalg.cholesky(G0 + root * IG)
            self.stats[m] = {"A": A0, "G": G0}
            self.factors[m] = {"LA": LA, "LG": LG, "in_dim": in_dim}
            self.eye_A[m] = IA
            self.eye_G[m] = IG

    def _register_hooks(self) -> None:
        self._hooks = []

        def fwd_hook(module: nn.Module, inputs, output):
            a = inputs[0].detach()
            self._cache[module]["a"] = a
            self._cache[module]["g"] = None
            if output.requires_grad:
                def store_grad(grad_out: torch.Tensor):
                    self._cache[module]["g"] = grad_out.detach()
                output.register_hook(store_grad)

        for m in self.modules:
            self._hooks.append(m.register_forward_hook(fwd_hook))

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @staticmethod
    def _safe_cholesky(M: torch.Tensor, max_tries: int = 6) -> torch.Tensor:
        I = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
        jitter = 0.0
        for k in range(max_tries):
            L, info = torch.linalg.cholesky_ex(M + jitter * I, check_errors=False)
            if int(info.item()) == 0:
                return L
            jitter = (10.0 ** k) * 1e-6
        return torch.linalg.cholesky(M + jitter * I)

    def _balanced_factor_damping(self, m: nn.Module) -> tuple[float, float]:
        if self.cfg.damping <= 0.0:
            return 0.0, 0.0
        A = self.stats[m]["A"]
        G = self.stats[m]["G"]
        meanA = max(torch.trace(A).item() / A.shape[0], 1e-12)
        meanG = max(torch.trace(G).item() / G.shape[0], 1e-12)
        pi = math.sqrt(meanA / meanG)
        root = math.sqrt(self.cfg.damping)
        return pi * root, root / pi

    @staticmethod
    def _sym(M: torch.Tensor) -> torch.Tensor:
        return 0.5 * (M + M.transpose(-2, -1))

    def _apply_gradient_momentum(self, m: nn.Module, grad_2d: torch.Tensor) -> torch.Tensor:
        momentum = float(self.cfg.momentum)
        if momentum <= 0.0:
            return grad_2d

        state = self.state[m]
        buf = state.get("momentum_buffer")
        if buf is None or buf.shape != grad_2d.shape or buf.device != grad_2d.device or buf.dtype != grad_2d.dtype:
            buf = torch.zeros_like(grad_2d)
            state["momentum_buffer"] = buf

        buf.mul_(momentum).add_(grad_2d)
        return grad_2d.add(buf, alpha=momentum) if self.cfg.nesterov else buf

    def _adjust_step_lr(self, lr: float, matrix_shape: tuple[int, int]) -> float:
        if self.cfg.lr_adjustment == "none":
            return lr
        return _adjust_muon_lr(lr, matrix_shape, mode=self.cfg.lr_adjustment)

    def _linear_reduce_factors(self, a: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = a.float().reshape(a.shape[0], -1, a.shape[-1])
        g = g.float().reshape(g.shape[0], -1, g.shape[-1])
        B = float(a.shape[0])
        a_bar = a.mean(dim=1)
        g_sum = g.sum(dim=1)
        A_b = (a_bar.transpose(0, 1) @ a_bar) / B
        G_b = (g_sum.transpose(0, 1) @ g_sum) * B
        return self._sym(A_b), self._sym(G_b)

    def _conv2d_reduce_factors(self, m: nn.Conv2d, a: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = F.unfold(
            a.float(),
            kernel_size=m.kernel_size,
            dilation=m.dilation,
            padding=m.padding,
            stride=m.stride,
        )
        B = float(patches.shape[0])
        a_bar = patches.transpose(1, 2).mean(dim=1)
        g = g.float().reshape(patches.shape[0], m.out_channels, -1).transpose(1, 2)
        g_sum = g.sum(dim=1)
        A_b = (a_bar.transpose(0, 1) @ a_bar) / B
        G_b = (g_sum.transpose(0, 1) @ g_sum) * B
        return self._sym(A_b), self._sym(G_b)

    @torch.no_grad()
    def maybe_update_stats(self) -> None:
        self._step += 1

        update_stats = (self.cfg.stats_update_every > 0 and self._step % self.cfg.stats_update_every == 0)
        update_factors = (self.cfg.factor_update_every > 0 and self._step % self.cfg.factor_update_every == 0)

        if update_stats:
            gamma = self.cfg.ema_decay
            one_minus = 1.0 - gamma
            for m in self.modules:
                a = self._cache[m].get("a", None)
                g = self._cache[m].get("g", None)
                if a is None or g is None or m.weight.grad is None:
                    continue
                if isinstance(m, nn.Linear):
                    A_b, G_b = self._linear_reduce_factors(a, g)
                elif isinstance(m, nn.Conv2d):
                    A_b, G_b = self._conv2d_reduce_factors(m, a, g)
                else:
                    raise TypeError(f"Unsupported module type: {type(m)}")
                self.stats[m]["A"].mul_(gamma).add_(A_b, alpha=one_minus)
                self.stats[m]["G"].mul_(gamma).add_(G_b, alpha=one_minus)

        if update_factors:
            for m in self.modules:
                a_damp, g_damp = self._balanced_factor_damping(m)
                A = self.stats[m]["A"] + a_damp * self.eye_A[m]
                G = self.stats[m]["G"] + g_damp * self.eye_G[m]
                self.factors[m]["LA"].copy_(self._safe_cholesky(A))
                self.factors[m]["LG"].copy_(self._safe_cholesky(G))

        for m in self.modules:
            self._cache[m]["g"] = None

    @torch.no_grad()
    def compute_steps(self, lr: float) -> list[tuple[nn.Parameter, torch.Tensor, float, float]]:
        groups = defaultdict(list)
        for m in self.modules:
            if m.weight.grad is None:
                continue
            grad_2d = m.weight.grad.reshape(m.weight.shape[0], -1).float()
            update_2d = self._apply_gradient_momentum(m, grad_2d)
            key = (update_2d.shape[0], update_2d.shape[1], update_2d.device, update_2d.dtype)
            groups[key].append((m, update_2d))

        steps_for_params: list[tuple[nn.Parameter, torch.Tensor, float, float]] = []

        for mods in groups.values():
            P_batch, LA_batch, LG_batch, meta = [], [], [], []
            for m, update_2d in mods:
                P_batch.append(update_2d)
                LA_batch.append(self.factors[m]["LA"])
                LG_batch.append(self.factors[m]["LG"])
                meta.append(m)

            P_batch = torch.stack(P_batch, dim=0).contiguous()
            LA_batch = torch.stack(LA_batch, dim=0).contiguous()
            LG_batch = torch.stack(LG_batch, dim=0).contiguous()

            tmp = torch.linalg.solve_triangular(LG_batch, P_batch, upper=False)
            P_hat = torch.linalg.solve_triangular(
                LA_batch, tmp.transpose(-2, -1), upper=False
            ).transpose(-2, -1)

            Q_hat = muon_quintic_ns(
                P_hat,
                ns_steps=self.cfg.muon_ns_steps,
                ns_coefficients=self.cfg.muon_ns_coefficients,
                eps=self.cfg.muon_ns_eps,
            )
            X_hat = -float(self.cfg.muon_eps) * Q_hat

            tmp2 = torch.linalg.solve_triangular(
                LG_batch.transpose(-2, -1), X_hat, upper=True
            )
            X_batch = torch.linalg.solve_triangular(
                LA_batch.transpose(-2, -1), tmp2.transpose(-2, -1), upper=True
            ).transpose(-2, -1)

            for i, m in enumerate(meta):
                X = X_batch[i].reshape_as(m.weight)
                step_lr = self._adjust_step_lr(lr, tuple(P_batch[i].shape))
                steps_for_params.append((m.weight, X, lr, step_lr))

        if self.cfg.max_step_norm is not None and steps_for_params:
            sq = torch.zeros((), device=steps_for_params[0][0].device, dtype=torch.float32)
            for _, step, _, _ in steps_for_params:
                sq += step.float().square().sum()
            norm = sq.sqrt()
            scale = min(1.0, float(self.cfg.max_step_norm) / float(norm + 1e-12))
            steps_for_params = [(p, s * scale, wd_lr, step_lr) for p, s, wd_lr, step_lr in steps_for_params]

        return steps_for_params

    @torch.no_grad()
    def apply_steps(self, steps_for_params: list[tuple[nn.Parameter, torch.Tensor, float, float]]) -> None:
        for p, step, weight_decay_lr, step_lr in steps_for_params:
            if self.cfg.weight_decay != 0.0:
                p.mul_(1.0 - weight_decay_lr * self.cfg.weight_decay)
            p.add_(step.to(dtype=p.dtype), alpha=step_lr)


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    mode: str = "muon"
    data_root: str = "./data/imagenette"
    download_imagenette: bool = False
    train_subset: Optional[int] = None
    val_subset: Optional[int] = None
    model_name: str = "tiny"
    steps: int = 2000
    eval_every: int = 100
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 8
    image_size: int = 224
    patch_size: int = 16
    seed: int = 1
    amp_bf16: bool = True
    lr: float = 2e-3
    aux_lr: Optional[float] = None
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 200
    warmup_fraction: Optional[float] = None
    lr_schedule: str = "cosine"
    flat_lr_fraction: float = 0.0
    min_lr_scale: float = 0.1
    grad_clip: Optional[float] = 1.0
    pool: str = "mean"
    separate_qkv: bool = True
    label_smoothing: float = 0.1
    train_crop_min_scale: float = 0.35
    random_erasing_prob: float = 0.0
    log_json: Optional[str] = None
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_eps: float = 1e-7
    muon_ns_steps: int = 5
    muon_lr_adjustment: str = "match_rms_adamw"
    kfac_damping: float = 1e-3
    kfac_ema_decay: float = 0.95
    kfac_stats_update_every: int = 20
    kfac_factor_update_every: int = 20
    kfac_momentum: float = 0.95
    kfac_nesterov: bool = True
    kfac_muon_eps: float = 0.1
    kfac_muon_ns_steps: int = 5
    kfac_muon_ns_eps: float = 1e-7
    kfac_muon_lr_adjustment: str = "none"
    kfac_max_step_norm: Optional[float] = None


def validate_train_config(cfg: TrainConfig) -> None:
    if cfg.eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {cfg.eval_every}")
    if cfg.steps <= 0:
        raise ValueError(f"steps must be positive, got {cfg.steps}")
    if cfg.image_size <= 0:
        raise ValueError(f"image_size must be positive, got {cfg.image_size}")
    if cfg.patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {cfg.patch_size}")
    if cfg.image_size % cfg.patch_size != 0:
        raise ValueError(
            f"image_size must be divisible by patch_size={cfg.patch_size}, got image_size={cfg.image_size}"
        )
    if not 0.0 <= cfg.muon_momentum < 1.0:
        raise ValueError(f"muon_momentum must be in [0, 1), got {cfg.muon_momentum}")
    if not 0.0 <= cfg.kfac_momentum < 1.0:
        raise ValueError(f"kfac_momentum must be in [0, 1), got {cfg.kfac_momentum}")
    if cfg.kfac_muon_lr_adjustment not in {"none", "original", "match_rms_adamw"}:
        raise ValueError(
            f"kfac_muon_lr_adjustment must be one of none/original/match_rms_adamw, got {cfg.kfac_muon_lr_adjustment}"
        )
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
    if not 0.0 < cfg.train_crop_min_scale <= 1.0:
        raise ValueError(f"train_crop_min_scale must be in (0, 1], got {cfg.train_crop_min_scale}")
    if not 0.0 <= cfg.random_erasing_prob <= 1.0:
        raise ValueError(f"random_erasing_prob must be in [0, 1], got {cfg.random_erasing_prob}")


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


def set_optimizer_lr(opt: Optional[torch.optim.Optimizer], lr: float) -> None:
    if opt is None:
        return
    for group in opt.param_groups:
        group["lr"] = lr


def train(cfg: TrainConfig) -> list[dict]:
    validate_train_config(cfg)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        train_crop_min_scale=cfg.train_crop_min_scale,
        random_erasing_prob=cfg.random_erasing_prob,
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"Train loader has zero batches. Increase train_subset or lower batch_size (batch_size={cfg.batch_size})."
        )
    train_iter = cycle(train_loader)

    model = build_vit(
        model_name=cfg.model_name,
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        num_classes=IMAGENETTE_NUM_CLASSES,
        pool=cfg.pool,
        separate_qkv=cfg.separate_qkv,
    ).to(device)
    muon_modules, muon_params, _aux_params = split_muon_and_aux(model, exclude_first_last=True)

    aux_lr = cfg.lr if cfg.aux_lr is None else cfg.aux_lr
    aux_param_groups = build_aux_adamw_param_groups(model, muon_params, weight_decay=cfg.weight_decay)
    aux_optim = torch.optim.AdamW(aux_param_groups, lr=aux_lr, betas=cfg.betas)

    if cfg.mode == "muon":
        muon_optim = FlattenedMuon(
            muon_params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=cfg.muon_momentum,
            nesterov=cfg.muon_nesterov,
            eps=cfg.muon_eps,
            ns_steps=cfg.muon_ns_steps,
            lr_adjustment=cfg.muon_lr_adjustment,
        )
        kfac = None
    elif cfg.mode == "kfac_muon":
        muon_optim = None
        kfac = KFACReduce(
            muon_modules,
            KFACConfig(
                damping=cfg.kfac_damping,
                ema_decay=cfg.kfac_ema_decay,
                stats_update_every=cfg.kfac_stats_update_every,
                factor_update_every=cfg.kfac_factor_update_every,
                momentum=cfg.kfac_momentum,
                nesterov=cfg.kfac_nesterov,
                muon_eps=cfg.kfac_muon_eps,
                muon_ns_steps=cfg.kfac_muon_ns_steps,
                muon_ns_eps=cfg.kfac_muon_ns_eps,
                lr_adjustment=cfg.kfac_muon_lr_adjustment,
                weight_decay=cfg.weight_decay,
                max_step_norm=cfg.kfac_max_step_norm,
            ),
        )
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    history: list[dict] = []
    wall_t0 = time.time()
    model.train()

    warmup_steps = resolve_warmup_steps(cfg.steps, cfg.warmup_steps, cfg.warmup_fraction)

    try:
        for step in range(1, cfg.steps + 1):
            lr_scale = lr_schedule_scale(
                step,
                cfg.steps,
                warmup_steps=warmup_steps,
                min_lr_scale=cfg.min_lr_scale,
                schedule=cfg.lr_schedule,
                flat_lr_fraction=cfg.flat_lr_fraction,
            )
            cur_muon_lr = cfg.lr * lr_scale
            cur_aux_lr = aux_lr * lr_scale

            set_optimizer_lr(aux_optim, cur_aux_lr)
            if muon_optim is not None:
                set_optimizer_lr(muon_optim, cur_muon_lr)

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

            if cfg.mode == "muon":
                assert muon_optim is not None
                aux_optim.step()
                muon_optim.step()
            else:
                assert kfac is not None
                kfac.maybe_update_stats()
                steps_for_params = kfac.compute_steps(cur_muon_lr)
                aux_optim.step()
                kfac.apply_steps(steps_for_params)

            if step == 1 or step % cfg.eval_every == 0 or step == cfg.steps:
                elapsed = time.time() - wall_t0
                batch_acc = accuracy_top1(logits.detach(), y)
                val_loss, val_acc = evaluate(model, val_loader, device, amp_bf16=cfg.amp_bf16)
                record = dict(
                    step=step,
                    time=elapsed,
                    muon_lr=float(cur_muon_lr),
                    aux_lr=float(cur_aux_lr),
                    lr_scale=float(lr_scale),
                    batch_loss=float(loss.detach()),
                    batch_acc=float(batch_acc),
                    val_loss=float(val_loss),
                    val_acc=float(val_acc),
                )
                history.append(record)
                print(
                    f"[{cfg.mode:10s}] step {step:5d}/{cfg.steps}  "
                    f"muon_lr {cur_muon_lr:.3e}  aux_lr {cur_aux_lr:.3e}  "
                    f"batch_loss {record['batch_loss']:.4f}  val_acc {100.0 * val_acc:5.2f}%  "
                    f"time {elapsed/60.0:6.1f} min"
                )
                if cfg.log_json is not None:
                    path = Path(cfg.log_json)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(history, indent=2))
    finally:
        if kfac is not None:
            kfac.close()
    return history


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Stage-1 ViT/Imagenette Muon vs reduce-KFAC-Muon benchmark")
    parser.add_argument("--mode", choices=["muon", "kfac_muon"], default="muon")
    parser.add_argument("--data-root", type=str, default="./data/imagenette")
    parser.add_argument("--download-imagenette", action="store_true")
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--model-name", choices=["tiny", "small"], default="tiny")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-amp-bf16", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--aux-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--warmup-fraction", type=float, default=None)
    parser.add_argument("--lr-schedule", choices=["cosine", "cosine_to_zero", "flat_cosine"], default="cosine")
    parser.add_argument("--flat-lr-fraction", type=float, default=0.0)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--pool", choices=["mean", "cls"], default="mean")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--train-crop-min-scale", type=float, default=0.35)
    parser.add_argument("--random-erasing-prob", type=float, default=0.0)
    parser.add_argument("--log-json", type=str, default=None)

    parser.set_defaults(muon_nesterov=True, kfac_nesterov=True, separate_qkv=True)
    parser.add_argument("--separate-qkv", action="store_true")
    parser.add_argument("--fused-qkv", dest="separate_qkv", action="store_false")

    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-nesterov", action="store_true")
    parser.add_argument("--no-muon-nesterov", dest="muon_nesterov", action="store_false")
    parser.add_argument("--muon-eps", type=float, default=1e-7)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-lr-adjustment", choices=["original", "match_rms_adamw"], default="match_rms_adamw")

    parser.add_argument("--kfac-damping", type=float, default=1e-3)
    parser.add_argument("--kfac-ema-decay", type=float, default=0.95)
    parser.add_argument("--kfac-stats-update-every", type=int, default=20)
    parser.add_argument("--kfac-factor-update-every", type=int, default=20)
    parser.add_argument("--kfac-momentum", type=float, default=0.95)
    parser.add_argument("--kfac-nesterov", action="store_true")
    parser.add_argument("--no-kfac-nesterov", dest="kfac_nesterov", action="store_false")
    parser.add_argument("--kfac-muon-eps", type=float, default=0.1)
    parser.add_argument("--kfac-muon-ns-steps", type=int, default=5)
    parser.add_argument("--kfac-muon-ns-eps", type=float, default=1e-7)
    parser.add_argument(
        "--kfac-muon-lr-adjustment",
        choices=["none", "original", "match_rms_adamw"],
        default="none",
    )
    parser.add_argument("--kfac-max-step-norm", type=float, default=None)

    args = parser.parse_args()
    return TrainConfig(
        mode=args.mode,
        data_root=args.data_root,
        download_imagenette=args.download_imagenette,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        model_name=args.model_name,
        steps=args.steps,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        patch_size=args.patch_size,
        seed=args.seed,
        amp_bf16=not args.no_amp_bf16,
        lr=args.lr,
        aux_lr=args.aux_lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        warmup_steps=args.warmup_steps,
        warmup_fraction=args.warmup_fraction,
        lr_schedule=args.lr_schedule,
        flat_lr_fraction=args.flat_lr_fraction,
        min_lr_scale=args.min_lr_scale,
        grad_clip=args.grad_clip,
        pool=args.pool,
        separate_qkv=args.separate_qkv,
        label_smoothing=args.label_smoothing,
        train_crop_min_scale=args.train_crop_min_scale,
        random_erasing_prob=args.random_erasing_prob,
        log_json=args.log_json,
        muon_momentum=args.muon_momentum,
        muon_nesterov=args.muon_nesterov,
        muon_eps=args.muon_eps,
        muon_ns_steps=args.muon_ns_steps,
        muon_lr_adjustment=args.muon_lr_adjustment,
        kfac_damping=args.kfac_damping,
        kfac_ema_decay=args.kfac_ema_decay,
        kfac_stats_update_every=args.kfac_stats_update_every,
        kfac_factor_update_every=args.kfac_factor_update_every,
        kfac_momentum=args.kfac_momentum,
        kfac_nesterov=args.kfac_nesterov,
        kfac_muon_eps=args.kfac_muon_eps,
        kfac_muon_ns_steps=args.kfac_muon_ns_steps,
        kfac_muon_ns_eps=args.kfac_muon_ns_eps,
        kfac_muon_lr_adjustment=args.kfac_muon_lr_adjustment,
        kfac_max_step_norm=args.kfac_max_step_norm,
    )


def main() -> None:
    cfg = parse_args()
    history = train(cfg)
    # if cfg.log_json is None:
    #     print(json.dumps(history[-5:], indent=2))


if __name__ == "__main__":
    main()
