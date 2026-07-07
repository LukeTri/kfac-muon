#!/usr/bin/env bash
set -euo pipefail

# VAST launcher for full ImageNet-1K experiments with timm train.py and ViT-B/16.
#
# Supports:
#   - MODE=muon
#   - MODE=kfac_muon
#
# Data setup:
#   PREP_IMAGENET1K=1 SOURCE_ROOT=/path/to/imagenet_source bash run_vast_imagenet1k_vitb16.sh
#
# SOURCE_ROOT should contain either:
#   train/<wnid>/*.JPEG and val/<wnid>/*.JPEG
#   OR official ILSVRC archives plus validation mapping; see prepare_imagenet1k_vast.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$SCRIPT_DIR}"
cd "$WORKDIR"

if [[ ! -f "train.py" ]]; then
  echo "train.py not found in: $WORKDIR"
  exit 1
fi

MODE="${MODE:-kfac_muon}"                      # muon | kfac_muon
DATA_ROOT="${DATA_ROOT:-/workspace/data/imagenet}"
SOURCE_ROOT="${SOURCE_ROOT:-/workspace/data/imagenet_source}"
PREP_IMAGENET1K="${PREP_IMAGENET1K:-1}"
PREP_SCRIPT="${PREP_SCRIPT:-${SCRIPT_DIR}/prepare_imagenet1k_vast.sh}"
PREP_MODE="${PREP_MODE:-symlink}"              # symlink | copy for extracted source trees

MODEL="${MODEL:-vit_base_patch16_224}"
NUM_CLASSES="${NUM_CLASSES:-1000}"
IMG_SIZE="${IMG_SIZE:-224}"
EPOCHS="${EPOCHS:-90}"
BATCH_SIZE="${BATCH_SIZE:-256}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-11}"

WEIGHT_DECAY="${WEIGHT_DECAY:-0.07}"
SCHED="${SCHED:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
MIN_LR="${MIN_LR:-1e-5}"
MIXUP="${MIXUP:-0.2}"
CUTMIX="${CUTMIX:-0.2}"
# For the 30-epoch pilot, set MIXUP_OFF_EPOCH=23 explicitly.
MIXUP_OFF_EPOCH="${MIXUP_OFF_EPOCH:-70}"
SMOOTHING="${SMOOTHING:-0.1}"
REPROB="${REPROB:-0.1}"
DROP_PATH="${DROP_PATH:-0.1}"

AMP="${AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"

if [[ -z "${LR:-}" ]]; then
  if [[ "$MODE" == "kfac_muon" || "$MODE" == "muon" ]]; then
    LR="1e-3"
  else
    echo "Unsupported MODE: $MODE (expected muon or kfac_muon)"
    exit 1
  fi
fi

OPT_BETA1="${OPT_BETA1:-0.9}"
OPT_BETA2="${OPT_BETA2:-0.95}"

KFAC_DAMPING="${KFAC_DAMPING:-5e-5}"
KFAC_MUON_EPS="${KFAC_MUON_EPS:-0.012}"
KFAC_MOMENTUM="${KFAC_MOMENTUM:-0.9}"
KFAC_NESTEROV="${KFAC_NESTEROV:-1}"
KFAC_STATS_UPDATE_EVERY="${KFAC_STATS_UPDATE_EVERY:-2}"
KFAC_FACTOR_UPDATE_EVERY="${KFAC_FACTOR_UPDATE_EVERY:-2}"
KFAC_MUON_LR_ADJUSTMENT="${KFAC_MUON_LR_ADJUSTMENT:-match_rms_adamw}"

OUTPUT="${OUTPUT:-/workspace/logs/timm_train}"
EXPERIMENT="${EXPERIMENT:-vitb16_in1k_${MODE}_e${EPOCHS}_lr${LR}_b${BATCH_SIZE}_seed${SEED}}"
LOG_INTERVAL="${LOG_INTERVAL:-200}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
CHECKPOINT_HIST="${CHECKPOINT_HIST:-2}"
# By default keep recent checkpoints so long VAST runs can be resumed.
# Set CHECKPOINT_FINAL_ONLY=1 to save only at the final epoch.
CHECKPOINT_FINAL_ONLY="${CHECKPOINT_FINAL_ONLY:-0}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$DATA_ROOT" "$OUTPUT"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> GPU:"
  nvidia-smi -L || true
fi

if [[ "$PREP_IMAGENET1K" == "1" ]]; then
  if [[ ! -f "$PREP_SCRIPT" ]]; then
    echo "Prep script not found: $PREP_SCRIPT"
    exit 1
  fi
  echo "==> Preparing ImageNet-1K dataset"
  SOURCE_ROOT="$SOURCE_ROOT" OUT_ROOT="$DATA_ROOT" MODE="$PREP_MODE" bash "$PREP_SCRIPT"
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

if [[ "$CHECKPOINT_FINAL_ONLY" == "1" ]]; then
  CMD+=(--checkpoint-final-only)
else
  CMD+=(--no-checkpoint-final-only)
fi

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
