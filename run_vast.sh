#!/usr/bin/env bash
set -euo pipefail

# Run-only Vast.ai launcher for Imagenette Muon/KFAC-Muon.
#
# Assumes:
#   1) repo is already cloned
#   2) dependencies are already installed
#
# Usage:
#   bash run_vast.sh
#
# Common overrides:
#   MODE="kfac_muon" \
#   STEPS="6000" \
#   LR="0.005" \
#   bash run_vast.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$SCRIPT_DIR}"
cd "$WORKDIR"

# Supports both current and older script naming.
SCRIPT_NAME="${SCRIPT_NAME:-vit_imagenette_muon_kfac_momentum.py}"
if [[ "$SCRIPT_NAME" == "vit_imagenette_stage1_muon_kfac_momentum.py" && ! -f "$SCRIPT_NAME" && -f "vit_imagenette_muon_kfac_momentum.py" ]]; then
  SCRIPT_NAME="vit_imagenette_muon_kfac_momentum.py"
fi
if [[ ! -f "$SCRIPT_NAME" ]]; then
  echo "Script not found: $SCRIPT_NAME"
  exit 1
fi

MODE="${MODE:-kfac_muon}"  # kfac_muon | muon
DATASET="${DATASET:-imagenette}"  # used only by the multi-dataset script
DATA_ROOT="${DATA_ROOT:-/workspace/data/imagenette}"
DOWNLOAD_IMAGENETTE="${DOWNLOAD_IMAGENETTE:-1}"
DOWNLOAD_CIFAR100="${DOWNLOAD_CIFAR100:-1}"

MODEL_NAME="${MODEL_NAME:-tiny}"
STEPS="${STEPS:-6000}"
EVAL_EVERY="${EVAL_EVERY:-250}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
MIN_LR_SCALE="${MIN_LR_SCALE:-0.04}"
POOL="${POOL:-mean}"
AUX_LR="${AUX_LR:-0.002}"

if [[ -z "${SEED:-}" ]]; then
  if [[ "$MODE" == "kfac_muon" ]]; then
    SEED="2"
  else
    SEED="1"
  fi
fi

if [[ -z "${LR:-}" ]]; then
  if [[ "$MODE" == "kfac_muon" ]]; then
    LR="0.005"
  else
    LR="0.001"
  fi
fi

KFAC_DAMPING="${KFAC_DAMPING:-0.01}"
KFAC_MUON_EPS="${KFAC_MUON_EPS:-0.1}"
KFAC_MOMENTUM="${KFAC_MOMENTUM:-0.95}"
KFAC_NESTEROV="${KFAC_NESTEROV:-1}"
KFAC_STATS_UPDATE_EVERY="${KFAC_STATS_UPDATE_EVERY:-1}"
KFAC_FACTOR_UPDATE_EVERY="${KFAC_FACTOR_UPDATE_EVERY:-1}"
KFAC_MUON_LR_ADJUSTMENT="${KFAC_MUON_LR_ADJUSTMENT:-original}"

LOG_JSON="${LOG_JSON:-/workspace/logs/${MODE}_imagenette.json}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$DATA_ROOT"
if [[ -n "$LOG_JSON" ]]; then
  mkdir -p "$(dirname "$LOG_JSON")"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> GPU:"
  nvidia-smi -L || true
fi

CMD=(python "$SCRIPT_NAME")

if [[ "$SCRIPT_NAME" == "vit_cifar100_imagenet_muon_kfac_momentum.py" ]]; then
  CMD+=(--dataset "$DATASET")
fi

CMD+=(
  --mode "$MODE"
  --data-root "$DATA_ROOT"
  --model-name "$MODEL_NAME"
  --steps "$STEPS"
  --eval-every "$EVAL_EVERY"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --lr "$LR"
  --aux-lr "$AUX_LR"
  --weight-decay "$WEIGHT_DECAY"
  --warmup-steps "$WARMUP_STEPS"
  --min-lr-scale "$MIN_LR_SCALE"
  --pool "$POOL"
  --seed "$SEED"
)

if [[ "$DOWNLOAD_IMAGENETTE" == "1" ]]; then
  CMD+=(--download-imagenette)
fi
if [[ "$SCRIPT_NAME" == "vit_cifar100_imagenet_muon_kfac_momentum.py" && "$DATASET" == "cifar100" && "$DOWNLOAD_CIFAR100" == "0" ]]; then
  CMD+=(--no-download-cifar100)
fi
if [[ -n "$LOG_JSON" ]]; then
  CMD+=(--log-json "$LOG_JSON")
fi

if [[ "$MODE" == "kfac_muon" ]]; then
  CMD+=(
    --kfac-damping "$KFAC_DAMPING"
    --kfac-muon-eps "$KFAC_MUON_EPS"
    --kfac-momentum "$KFAC_MOMENTUM"
    --kfac-stats-update-every "$KFAC_STATS_UPDATE_EVERY"
    --kfac-factor-update-every "$KFAC_FACTOR_UPDATE_EVERY"
    --kfac-muon-lr-adjustment "$KFAC_MUON_LR_ADJUSTMENT"
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
