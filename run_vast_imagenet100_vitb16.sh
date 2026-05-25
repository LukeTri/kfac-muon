#!/usr/bin/env bash
set -euo pipefail

# VAST launcher for ImageNet-100 experiments with timm train.py and ViT-B/16.
#
# Supports:
#   - MODE=muon
#   - MODE=kfac_muon
#
# Example:
#   MODE=kfac_muon \
#   EPOCHS=90 \
#   BATCH_SIZE=64 \
#   SEED=11 \
#   bash run_vast_imagenet100_vitb16.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$SCRIPT_DIR}"
cd "$WORKDIR"

if [[ ! -f "train.py" ]]; then
  echo "train.py not found in: $WORKDIR"
  exit 1
fi

MODE="${MODE:-kfac_muon}"                      # muon | kfac_muon
DATA_ROOT="${DATA_ROOT:-/workspace/data/imagenet100}"
DOWNLOAD_IMAGENET100="${DOWNLOAD_IMAGENET100:-1}"
DOWNLOAD_SCRIPT="${DOWNLOAD_SCRIPT:-${SCRIPT_DIR}/download_imagenet100_vast.sh}"

MODEL="${MODEL:-vit_base_patch16_224}"
NUM_CLASSES="${NUM_CLASSES:-100}"
IMG_SIZE="${IMG_SIZE:-224}"
EPOCHS="${EPOCHS:-90}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-11}"

WEIGHT_DECAY="${WEIGHT_DECAY:-0.07}"
SCHED="${SCHED:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
MIN_LR="${MIN_LR:-1e-5}"
MIXUP="${MIXUP:-0.2}"
CUTMIX="${CUTMIX:-0.2}"
MIXUP_OFF_EPOCH="${MIXUP_OFF_EPOCH:-70}"
SMOOTHING="${SMOOTHING:-0.1}"
REPROB="${REPROB:-0.1}"
DROP_PATH="${DROP_PATH:-0.1}"

AMP="${AMP:-1}"                                # 1 => --amp
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"

if [[ -z "${LR:-}" ]]; then
  if [[ "$MODE" == "kfac_muon" ]]; then
    LR="5.5e-4"
  elif [[ "$MODE" == "muon" ]]; then
    LR="5.5e-4"
  else
    echo "Unsupported MODE: $MODE (expected muon or kfac_muon)"
    exit 1
  fi
fi

OPT_BETA1="${OPT_BETA1:-0.9}"
OPT_BETA2="${OPT_BETA2:-0.95}"

KFAC_DAMPING="${KFAC_DAMPING:-1.5e-4}"
KFAC_MUON_EPS="${KFAC_MUON_EPS:-0.038}"
KFAC_MOMENTUM="${KFAC_MOMENTUM:-0.9}"
KFAC_NESTEROV="${KFAC_NESTEROV:-1}"
KFAC_STATS_UPDATE_EVERY="${KFAC_STATS_UPDATE_EVERY:-2}"
KFAC_FACTOR_UPDATE_EVERY="${KFAC_FACTOR_UPDATE_EVERY:-2}"
KFAC_MUON_LR_ADJUSTMENT="${KFAC_MUON_LR_ADJUSTMENT:-match_rms_adamw}"

OUTPUT="${OUTPUT:-/workspace/logs/timm_train}"
EXPERIMENT="${EXPERIMENT:-vitb16_in100_${MODE}_e${EPOCHS}_lr${LR}_seed${SEED}}"
LOG_INTERVAL="${LOG_INTERVAL:-200}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
CHECKPOINT_HIST="${CHECKPOINT_HIST:-2}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$DATA_ROOT" "$OUTPUT"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> GPU:"
  nvidia-smi -L || true
fi

if [[ "$DOWNLOAD_IMAGENET100" == "1" ]]; then
  if [[ ! -f "$DOWNLOAD_SCRIPT" ]]; then
    echo "Download script not found: $DOWNLOAD_SCRIPT"
    exit 1
  fi
  echo "==> Preparing ImageNet-100 dataset"
  OUT_ROOT="$DATA_ROOT" bash "$DOWNLOAD_SCRIPT"
fi

CMD=(
  python3 train.py
  --data-dir "$DATA_ROOT"
  --dataset image_folder
  --train-split train
  --val-split val
  --num-classes "$NUM_CLASSES"
  --model "$MODEL"
  --img-size "$IMG_SIZE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --validation-batch-size "$VAL_BATCH_SIZE"
  --workers "$WORKERS"
  --opt "$MODE"
  --lr "$LR"
  --opt-betas "$OPT_BETA1" "$OPT_BETA2"
  --weight-decay "$WEIGHT_DECAY"
  --sched "$SCHED"
  --warmup-epochs "$WARMUP_EPOCHS"
  --min-lr "$MIN_LR"
  --mixup "$MIXUP"
  --cutmix "$CUTMIX"
  --mixup-off-epoch "$MIXUP_OFF_EPOCH"
  --smoothing "$SMOOTHING"
  --reprob "$REPROB"
  --drop-path "$DROP_PATH"
  --seed "$SEED"
  --log-interval "$LOG_INTERVAL"
  --val-interval "$VAL_INTERVAL"
  --checkpoint-hist "$CHECKPOINT_HIST"
  --output "$OUTPUT"
  --experiment "$EXPERIMENT"
)

if [[ "$AMP" == "1" ]]; then
  CMD+=(--amp --amp-dtype "$AMP_DTYPE")
fi

if [[ "$MODE" == "kfac_muon" ]]; then
  CMD+=(
    --kfac-damping "$KFAC_DAMPING"
    --kfac-muon-eps "$KFAC_MUON_EPS"
    --kfac-momentum "$KFAC_MOMENTUM"
    --kfac-stats-update-every "$KFAC_STATS_UPDATE_EVERY"
    --kfac-factor-update-every "$KFAC_FACTOR_UPDATE_EVERY"
    --kfac-muon-lr-adjustment "$KFAC_MUON_LR_ADJUSTMENT"
    --kfac-aux-no-decay
  )
  if [[ "$KFAC_NESTEROV" == "1" ]]; then
    CMD+=(--kfac-nesterov)
  else
    CMD+=(--no-kfac-nesterov)
  fi
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_SPLIT=($EXTRA_ARGS)
  CMD+=("${EXTRA_SPLIT[@]}")
fi

echo "==> Running command:"
printf ' %q' "${CMD[@]}"
echo

exec "${CMD[@]}"
