#!/usr/bin/env bash
set -euo pipefail

# Prepare full ImageNet-1K on VAST in timm/ImageFolder layout.
#
# This script does not download ImageNet-1K. Put either an extracted ImageFolder
# tree or official ILSVRC/Kaggle archives in SOURCE_ROOT, then run this script.
#
# Supported sources:
#   SOURCE_ROOT/train/<wnid>/*.JPEG and SOURCE_ROOT/val/<wnid>/*.JPEG
#   OR
#   SOURCE_ROOT/ILSVRC2012_img_train.tar
#   SOURCE_ROOT/ILSVRC2012_img_val.tar
#   plus SOURCE_ROOT/LOC_val_solution.csv or SOURCE_ROOT/val_wnids.txt
#
# Output:
#   OUT_ROOT/train/<wnid>/*.JPEG
#   OUT_ROOT/val/<wnid>/*.JPEG

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-/workspace/data/imagenet_source}"
OUT_ROOT="${OUT_ROOT:-/workspace/data/imagenet}"
MODE="${MODE:-symlink}"                  # symlink | copy, for already-extracted source trees
OVERWRITE="${OVERWRITE:-0}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"
STRICT="${STRICT:-0}"
VAL_WNIDS_FILE="${VAL_WNIDS_FILE:-}"
REPORT_EVERY="${REPORT_EVERY:-50}"

mkdir -p "${OUT_ROOT}"

CMD=(
  python3 "${SCRIPT_DIR}/prepare_imagenet1k_vast.py"
  --source-root "${SOURCE_ROOT}"
  --out-root "${OUT_ROOT}"
  --mode "${MODE}"
  --report-every "${REPORT_EVERY}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi
if [[ "${VALIDATE_ONLY}" == "1" ]]; then
  CMD+=(--validate-only)
fi
if [[ "${STRICT}" == "1" ]]; then
  CMD+=(--strict)
fi
if [[ -n "${VAL_WNIDS_FILE}" ]]; then
  CMD+=(--val-wnids-file "${VAL_WNIDS_FILE}")
fi

echo "==> Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "[done] Use with timm train.py:"
echo "python train.py --data-dir ${OUT_ROOT} --dataset image_folder --train-split train --val-split val --num-classes 1000 ..."
