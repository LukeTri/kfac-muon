#!/usr/bin/env bash
set -euo pipefail

# Local-neighborhood sweep around current best CIFAR-100 ViT-S/16 settings.
# Focus:
# - KFAC around best rho settings (base + wide) with local lr/damping tweaks
# - A few KFAC runs adapted toward Muon's higher base lr with eps adjustment
#
# Usage:
#   PREP_DATA=1 bash run_vast_cifar100_local_best_sweep.sh
#   DRY_RUN=1 bash run_vast_cifar100_local_best_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$SCRIPT_DIR}"
cd "$WORKDIR"

if [[ ! -f "train.py" ]]; then
  echo "train.py not found in: $WORKDIR"
  exit 1
fi

PREP_DATA="${PREP_DATA:-0}"                  # 1 => run dataset prep
DRY_RUN="${DRY_RUN:-0}"                      # 1 => print commands only
DATA_ROOT="${DATA_ROOT:-/dev/shm/cifar100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/logs/timm_train}"

EPOCHS="${EPOCHS:-75}"
SEEDS="${SEEDS:-12}"                         # space-separated

MODEL="${MODEL:-vit_small_patch16_224}"
NUM_CLASSES="${NUM_CLASSES:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
WORKERS="${WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
LOG_INTERVAL="${LOG_INTERVAL:-200}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
CHECKPOINT_HIST="${CHECKPOINT_HIST:-2}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> GPU:"
  nvidia-smi -L || true
fi

if [[ "$PREP_DATA" == "1" ]]; then
  echo "==> Preparing CIFAR-100 at: $DATA_ROOT"
  OUT_ROOT="$DATA_ROOT" bash "$SCRIPT_DIR/download_cifar100_vast.sh"
fi

mkdir -p "$OUTPUT_ROOT"

run_train() {
  local exp="$1"
  local seed="$2"
  shift 2
  local extra_args=("$@")

  local cmd=(
    python3 train.py
    --data-dir "$DATA_ROOT"
    --dataset image_folder
    --train-split train
    --val-split val
    --num-classes "$NUM_CLASSES"
    --model "$MODEL"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --validation-batch-size "$VAL_BATCH_SIZE"
    --workers "$WORKERS"
    --weight-decay 0.07
    --sched cosine
    --min-lr 1e-5
    --mixup 0.2
    --cutmix 0.2
    --mixup-off-epoch 56
    --smoothing 0.1
    --reprob 0.1
    --drop-path 0.1
    --opt-betas 0.9 0.95
    --seed "$seed"
    --log-interval "$LOG_INTERVAL"
    --val-interval "$VAL_INTERVAL"
    --checkpoint-hist "$CHECKPOINT_HIST"
    --output "$OUTPUT_ROOT"
    --experiment "$exp"
    --amp
    --amp-dtype "$AMP_DTYPE"
  )
  cmd+=("${extra_args[@]}")

  echo
  echo "==> Running: $exp (seed=$seed)"
  printf ' %q' "${cmd[@]}"
  echo

  if [[ "$DRY_RUN" != "1" ]]; then
    "${cmd[@]}"
  fi
}

for seed in $SEEDS; do
  # KFAC around the two best families: base and rho_wide
  run_train "vits16_c100_kfac_base_refit_e${EPOCHS}_s${seed}" "$seed" \
    --opt kfac_muon --lr 5.5e-4 --warmup-epochs 8 \
    --kfac-damping 1.5e-4 --kfac-momentum 0.9 --kfac-nesterov \
    --kfac-muon-eps 0.038 --kfac-muon-lr-adjustment match_rms_adamw \
    --kfac-stats-update-every 2 --kfac-factor-update-every 2 \
    --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
    --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
    --kfac-lm-rho-low 1.2 --kfac-lm-rho-high 1.8 \
    --kfac-aux-no-decay

  run_train "vits16_c100_kfac_rho_wide_refit_e${EPOCHS}_s${seed}" "$seed" \
    --opt kfac_muon --lr 5.5e-4 --warmup-epochs 8 \
    --kfac-damping 1.5e-4 --kfac-momentum 0.9 --kfac-nesterov \
    --kfac-muon-eps 0.038 --kfac-muon-lr-adjustment match_rms_adamw \
    --kfac-stats-update-every 2 --kfac-factor-update-every 2 \
    --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
    --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
    --kfac-lm-rho-low 1.0 --kfac-lm-rho-high 2.0 \
    --kfac-aux-no-decay

  run_train "vits16_c100_kfac_rho_1p1_1p9_e${EPOCHS}_s${seed}" "$seed" \
    --opt kfac_muon --lr 5.5e-4 --warmup-epochs 8 \
    --kfac-damping 1.5e-4 --kfac-momentum 0.9 --kfac-nesterov \
    --kfac-muon-eps 0.038 --kfac-muon-lr-adjustment match_rms_adamw \
    --kfac-stats-update-every 2 --kfac-factor-update-every 2 \
    --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
    --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
    --kfac-lm-rho-low 1.1 --kfac-lm-rho-high 1.9 \
    --kfac-aux-no-decay

  run_train "vits16_c100_kfac_rho_0p9_2p1_e${EPOCHS}_s${seed}" "$seed" \
    --opt kfac_muon --lr 5.5e-4 --warmup-epochs 8 \
    --kfac-damping 1.5e-4 --kfac-momentum 0.9 --kfac-nesterov \
    --kfac-muon-eps 0.038 --kfac-muon-lr-adjustment match_rms_adamw \
    --kfac-stats-update-every 2 --kfac-factor-update-every 2 \
    --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
    --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
    --kfac-lm-rho-low 0.9 --kfac-lm-rho-high 2.1 \
    --kfac-aux-no-decay

  run_train "vits16_c100_kfac_rho_wide_damp12e5_e${EPOCHS}_s${seed}" "$seed" \
    --opt kfac_muon --lr 5.5e-4 --warmup-epochs 8 \
    --kfac-damping 1.2e-4 --kfac-momentum 0.9 --kfac-nesterov \
    --kfac-muon-eps 0.038 --kfac-muon-lr-adjustment match_rms_adamw \
    --kfac-stats-update-every 2 --kfac-factor-update-every 2 \
    --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
    --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
    --kfac-lm-rho-low 1.0 --kfac-lm-rho-high 2.0 \
    --kfac-aux-no-decay

  run_train "vits16_c100_kfac_rho_wide_damp18e5_e${EPOCHS}_s${seed}" "$seed" \
    --opt kfac_muon --lr 5.5e-4 --warmup-epochs 8 \
    --kfac-damping 1.8e-4 --kfac-momentum 0.9 --kfac-nesterov \
    --kfac-muon-eps 0.038 --kfac-muon-lr-adjustment match_rms_adamw \
    --kfac-stats-update-every 2 --kfac-factor-update-every 2 \
    --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
    --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
    --kfac-lm-rho-low 1.0 --kfac-lm-rho-high 2.0 \
    --kfac-aux-no-decay

  # Adapt KFAC toward Muon best base lr (with/without eps bump).
  run_train "vits16_c100_kfac_rho_wide_lr1e3_eps038_e${EPOCHS}_s${seed}" "$seed" \
    --opt kfac_muon --lr 1e-3 --warmup-epochs 8 \
    --kfac-damping 1.5e-4 --kfac-momentum 0.9 --kfac-nesterov \
    --kfac-muon-eps 0.038 --kfac-muon-lr-adjustment match_rms_adamw \
    --kfac-stats-update-every 2 --kfac-factor-update-every 2 \
    --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
    --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
    --kfac-lm-rho-low 1.0 --kfac-lm-rho-high 2.0 \
    --kfac-aux-no-decay

  run_train "vits16_c100_kfac_rho_wide_lr1e3_eps055_e${EPOCHS}_s${seed}" "$seed" \
    --opt kfac_muon --lr 1e-3 --warmup-epochs 8 \
    --kfac-damping 1.5e-4 --kfac-momentum 0.9 --kfac-nesterov \
    --kfac-muon-eps 0.055 --kfac-muon-lr-adjustment match_rms_adamw \
    --kfac-stats-update-every 2 --kfac-factor-update-every 2 \
    --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
    --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
    --kfac-lm-rho-low 1.0 --kfac-lm-rho-high 2.0 \
    --kfac-aux-no-decay
done

echo
echo "Local-best neighborhood sweep complete."
