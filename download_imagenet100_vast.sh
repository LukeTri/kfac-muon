#!/usr/bin/env bash
set -euo pipefail

# Prepare ImageNet-100 on VAST from Hugging Face into ImageFolder layout.
#
# Output:
#   $OUT_ROOT/train/<class>/*.jpg
#   $OUT_ROOT/val/<class>/*.jpg

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${OUT_ROOT:-/workspace/data/imagenet100}"
CACHE_DIR="${CACHE_DIR:-/workspace/.cache/huggingface}"
DATASET_ID="${DATASET_ID:-clane9/imagenet-100}"
OVERWRITE_SPLIT="${OVERWRITE_SPLIT:-0}"

mkdir -p "${OUT_ROOT}" "${CACHE_DIR}"

CMD=(
  python3 "${SCRIPT_DIR}/download_imagenet100_vast.py"
  --dataset-id "${DATASET_ID}"
  --out-root "${OUT_ROOT}"
  --cache-dir "${CACHE_DIR}"
)

if [[ "${OVERWRITE_SPLIT}" == "1" ]]; then
  CMD+=(--overwrite-split)
fi

echo "==> Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "[done] Use with timm train.py:"
echo "python train.py --data-dir ${OUT_ROOT} --dataset image_folder --train-split train --val-split val ..."
