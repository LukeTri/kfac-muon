#!/usr/bin/env bash
set -euo pipefail

# One-command Food-101 setup for timm train.py.
#
# Output layout:
#   $OUT_ROOT/train/<class>/*.jpg
#   $OUT_ROOT/val/<class>/*.jpg

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOURCE_ROOT="${SOURCE_ROOT:-/workspace/data/torchvision}"
OUT_ROOT="${OUT_ROOT:-/workspace/data/food101}"
LINK_MODE="${LINK_MODE:-symlink}"   # symlink | hardlink | copy
OVERWRITE="${OVERWRITE:-0}"         # 1 to rebuild train/val folders
REPORT_EVERY="${REPORT_EVERY:-5000}"

CMD=(
  python3 "${SCRIPT_DIR}/download_food101_vast.py"
  --source-root "${SOURCE_ROOT}"
  --out-root "${OUT_ROOT}"
  --link-mode "${LINK_MODE}"
  --report-every "${REPORT_EVERY}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "==> Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo
echo "[done] Run training with:"
echo "python train.py --data-dir ${OUT_ROOT} --dataset image_folder --train-split train --val-split val --num-classes 101 ..."
