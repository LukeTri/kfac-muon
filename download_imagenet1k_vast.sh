#!/usr/bin/env bash
set -euo pipefail

# Download + prepare ImageNet-1k on a VAST instance for timm train.py.
#
# Final layout:
#   $DATA_ROOT/train/<class>/*.JPEG
#   $DATA_ROOT/val/<class>/*.JPEG
#
# Usage:
#   1) Provide authenticated URLs (official ImageNet downloads), optionally cookies:
#      IMAGENET_TRAIN_URL="https://..." \
#      IMAGENET_VAL_URL="https://..." \
#      IMAGENET_DEVKIT_URL="https://..." \
#      IMAGENET_COOKIES="/path/to/cookies.txt" \
#      bash download_imagenet1k_vast.sh
#
#   2) If tarballs already exist at DATA_ROOT, just prepare:
#      PREPARE_ONLY=1 bash download_imagenet1k_vast.sh
#
# Optional env vars:
#   DATA_ROOT=/workspace/data/imagenet   (default)
#   KEEP_ARCHIVES=1                      (default: 1)
#   PREPARE_ONLY=0                       (default: 0)

DATA_ROOT="${DATA_ROOT:-/workspace/data/imagenet}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-1}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

TRAIN_TAR="${DATA_ROOT}/ILSVRC2012_img_train.tar"
VAL_TAR="${DATA_ROOT}/ILSVRC2012_img_val.tar"
DEVKIT_TAR="${DATA_ROOT}/ILSVRC2012_devkit_t12.tar.gz"

IMAGENET_TRAIN_URL="${IMAGENET_TRAIN_URL:-}"
IMAGENET_VAL_URL="${IMAGENET_VAL_URL:-}"
IMAGENET_DEVKIT_URL="${IMAGENET_DEVKIT_URL:-}"
IMAGENET_COOKIES="${IMAGENET_COOKIES:-}"

mkdir -p "${DATA_ROOT}"

download_if_missing() {
  local url="$1"
  local out="$2"
  if [[ -f "${out}" ]]; then
    echo "[skip] ${out} already exists"
    return
  fi
  if [[ -z "${url}" ]]; then
    echo "[error] Missing URL for ${out}"
    exit 1
  fi
  echo "[download] ${out}"
  if [[ -n "${IMAGENET_COOKIES}" ]]; then
    curl -L --fail --continue-at - --cookie "${IMAGENET_COOKIES}" -o "${out}" "${url}"
  else
    curl -L --fail --continue-at - -o "${out}" "${url}"
  fi
}

if [[ "${PREPARE_ONLY}" != "1" ]]; then
  command -v curl >/dev/null 2>&1 || { echo "[error] curl not found"; exit 1; }
  download_if_missing "${IMAGENET_TRAIN_URL}" "${TRAIN_TAR}"
  download_if_missing "${IMAGENET_VAL_URL}" "${VAL_TAR}"
  download_if_missing "${IMAGENET_DEVKIT_URL}" "${DEVKIT_TAR}"
fi

if [[ ! -f "${TRAIN_TAR}" || ! -f "${VAL_TAR}" || ! -f "${DEVKIT_TAR}" ]]; then
  echo "[error] Missing required archives in ${DATA_ROOT}:"
  echo "  - ILSVRC2012_img_train.tar"
  echo "  - ILSVRC2012_img_val.tar"
  echo "  - ILSVRC2012_devkit_t12.tar.gz"
  exit 1
fi

echo "[prepare] Checking Python deps"
python3 - <<'PY'
import importlib.util
missing = [m for m in ("torch", "torchvision", "scipy") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(
        "Missing Python deps: " + ", ".join(missing) +
        ". Install with: pip install torch torchvision scipy"
    )
print("Deps ok")
PY

echo "[prepare] Extracting train/val folders via torchvision (can take a while)"
DATA_ROOT="${DATA_ROOT}" python3 - <<'PY'
import os
from torchvision.datasets import ImageNet

root = os.environ["DATA_ROOT"]

# torchvision will parse devkit + extract train/val into class folders.
ImageNet(root=root, split="train")
ImageNet(root=root, split="val")

print("Prepared ImageNet at:", root)
print("Expected folders:")
print(" -", os.path.join(root, "train"))
print(" -", os.path.join(root, "val"))
PY

if [[ "${KEEP_ARCHIVES}" != "1" ]]; then
  echo "[cleanup] Removing original archives"
  rm -f "${TRAIN_TAR}" "${VAL_TAR}" "${DEVKIT_TAR}"
fi

echo "[done] You can now train with:"
echo "python train.py --data-dir ${DATA_ROOT} --dataset image_folder --train-split train --val-split val ..."
