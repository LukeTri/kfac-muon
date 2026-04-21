#!/usr/bin/env bash
set -euo pipefail

# One-command Places365 setup for timm train.py.
#
# Output layout:
#   $OUT_ROOT/train/<class>/*
#   $OUT_ROOT/val/<class>/*
#
# Defaults:
#   - train split: train-standard
#   - val split: val
#   - small images (256px)
#   - symlink materialization

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOURCE_ROOT="${SOURCE_ROOT:-/workspace/data/places365_raw}"
OUT_ROOT="${OUT_ROOT:-/workspace/data/places365}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train-standard}"
VAL_SPLIT="${VAL_SPLIT:-val}"
LINK_MODE="${LINK_MODE:-symlink}"      # symlink | hardlink | copy
USE_LARGE="${USE_LARGE:-0}"            # 1 => large images
OVERWRITE="${OVERWRITE:-0}"            # 1 => rebuild train/val folders
REPORT_EVERY="${REPORT_EVERY:-20000}"

CMD=(
  python3 "${SCRIPT_DIR}/download_places365_vast.py"
  --source-root "${SOURCE_ROOT}"
  --out-root "${OUT_ROOT}"
  --train-split "${TRAIN_SPLIT}"
  --val-split "${VAL_SPLIT}"
  --link-mode "${LINK_MODE}"
  --report-every "${REPORT_EVERY}"
)

if [[ "${USE_LARGE}" == "1" ]]; then
  CMD+=(--large)
else
  CMD+=(--small)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "==> Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo
echo "[done] Run training with:"
echo "python train.py --data-dir ${OUT_ROOT} --dataset image_folder --train-split train --val-split val --num-classes 365 ..."
