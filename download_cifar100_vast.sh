#!/usr/bin/env bash
set -euo pipefail

# One-command CIFAR-100 setup for timm train.py.
#
# Output layout:
#   $OUT_ROOT/train/<class>/*.png
#   $OUT_ROOT/val/<class>/*.png

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOURCE_ROOT="${SOURCE_ROOT:-/workspace/data/torchvision}"
OUT_ROOT="${OUT_ROOT:-/workspace/data/cifar100}"
LINK_MODE="${LINK_MODE:-copy}"      # accepted for CLI parity; PNG files are always written
OVERWRITE="${OVERWRITE:-0}"         # 1 to rebuild train/val folders
REPORT_EVERY="${REPORT_EVERY:-5000}"

CMD=(
  python3 "${SCRIPT_DIR}/download_cifar100_vast.py"
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
echo "python train.py --data-dir ${OUT_ROOT} --dataset image_folder --train-split train --val-split val --num-classes 100 ..."
