#!/usr/bin/env bash
set -euo pipefail

# CIFAR-100 ViT-S/16 sweep launcher for VAST.
#
# Goals:
# - PHASE=pilot: quick ranking sweep (single seed, shorter training).
# - PHASE=full: confirmation sweep (multi-seed, full training length).
#
# Usage examples:
#   PHASE=pilot PREP_DATA=1 bash run_vast_cifar100_vits16_sweep.sh
#   PHASE=full CASES="muon_fixed,kfac_vit_base,kfac_wd005" SEEDS="11 12 13" bash run_vast_cifar100_vits16_sweep.sh
#   DRY_RUN=1 PHASE=pilot bash run_vast_cifar100_vits16_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$SCRIPT_DIR}"
cd "$WORKDIR"

if [[ ! -f "train.py" ]]; then
  echo "train.py not found in: $WORKDIR"
  exit 1
fi

PHASE="${PHASE:-pilot}"                      # pilot | full
PREP_DATA="${PREP_DATA:-1}"                  # 1 => run dataset prep
DRY_RUN="${DRY_RUN:-0}"                      # 1 => print commands only
DATA_ROOT="${DATA_ROOT:-/dev/shm/cifar100}"  # matches your previous runs
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/logs/timm_train}"

MODEL="${MODEL:-vit_small_patch16_224}"
NUM_CLASSES="${NUM_CLASSES:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
WORKERS="${WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
LOG_INTERVAL="${LOG_INTERVAL:-200}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
CHECKPOINT_HIST="${CHECKPOINT_HIST:-2}"

# ViT-close defaults from prior successful settings.
MUON_LR_FIXED="${MUON_LR_FIXED:-6e-4}"
KFAC_LR_FIXED="${KFAC_LR_FIXED:-5.5e-4}"
KFAC_DAMPING_BASE="${KFAC_DAMPING_BASE:-1.5e-4}"
KFAC_EPS_BASE="${KFAC_EPS_BASE:-0.038}"
KFAC_MOMENTUM_BASE="${KFAC_MOMENTUM_BASE:-0.9}"
KFAC_STATS_EVERY_BASE="${KFAC_STATS_EVERY_BASE:-2}"
KFAC_FACTOR_EVERY_BASE="${KFAC_FACTOR_EVERY_BASE:-2}"
KFAC_LR_ADJUSTMENT_BASE="${KFAC_LR_ADJUSTMENT_BASE:-match_rms_adamw}"
MUON_LR_LOW="${MUON_LR_LOW:-5e-4}"
MUON_LR_HIGH="${MUON_LR_HIGH:-8e-4}"
KFAC_LR_LOW="${KFAC_LR_LOW:-4.5e-4}"
KFAC_LR_HIGH="${KFAC_LR_HIGH:-7e-4}"

if [[ "$PHASE" == "pilot" ]]; then
  EPOCHS="${EPOCHS:-75}"
  BASE_MIXUP_OFF_EPOCH="${BASE_MIXUP_OFF_EPOCH:-56}"
  DEFAULT_SEEDS="${DEFAULT_SEEDS:-12}"
  DEFAULT_CASES="muon_fixed,muon_lr_low,muon_lr_high,kfac_vit_base,kfac_lr_low,kfac_lr_high,kfac_lmoff,kfac_rho_wide,kfac_damp_5e5,kfac_eps0055,kfac_dp005,kfac_wd005"
elif [[ "$PHASE" == "full" ]]; then
  EPOCHS="${EPOCHS:-200}"
  BASE_MIXUP_OFF_EPOCH="${BASE_MIXUP_OFF_EPOCH:-140}"
  DEFAULT_SEEDS="${DEFAULT_SEEDS:-11 12 13}"
  DEFAULT_CASES="muon_fixed,kfac_vit_base,kfac_wd005"
else
  echo "Unsupported PHASE: $PHASE (expected pilot or full)"
  exit 1
fi

CASES="${CASES:-$DEFAULT_CASES}"             # comma-separated case names
SEEDS_STR="${SEEDS:-$DEFAULT_SEEDS}"         # space-separated seeds

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> GPU:"
  nvidia-smi -L || true
fi

if [[ "$PREP_DATA" == "1" ]]; then
  echo "==> Preparing CIFAR-100 at: $DATA_ROOT"
  OUT_ROOT="$DATA_ROOT" bash "$SCRIPT_DIR/download_cifar100_vast.sh"
fi

mkdir -p "$OUTPUT_ROOT"

contains_case() {
  local needle="$1"
  local list=",$CASES,"
  [[ "$list" == *",$needle,"* ]]
}

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
    --warmup-epochs 8
    --min-lr 1e-5
    --mixup 0.2
    --cutmix 0.2
    --mixup-off-epoch "$BASE_MIXUP_OFF_EPOCH"
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

for seed in $SEEDS_STR; do
  # Muon anchor and LR bracket.
  if contains_case "muon_fixed"; then
    run_train "vits16_c100_muon_fixed_e${EPOCHS}_s${seed}" "$seed" \
      --opt muon --lr "$MUON_LR_FIXED" \
      --warmup-epochs 10
  fi

  if contains_case "muon_lr_low"; then
    run_train "vits16_c100_muon_lr_low_e${EPOCHS}_s${seed}" "$seed" \
      --opt muon --lr "$MUON_LR_LOW" \
      --warmup-epochs 10
  fi

  if contains_case "muon_lr_high"; then
    run_train "vits16_c100_muon_lr_high_e${EPOCHS}_s${seed}" "$seed" \
      --opt muon --lr "$MUON_LR_HIGH" \
      --warmup-epochs 10
  fi

  # ViT-close KFAC baseline from prior experiments.
  if contains_case "kfac_vit_base"; then
    run_train "vits16_c100_kfac_vit_base_e${EPOCHS}_s${seed}" "$seed" \
      --opt kfac_muon --lr "$KFAC_LR_FIXED" \
      --kfac-damping "$KFAC_DAMPING_BASE" --kfac-momentum "$KFAC_MOMENTUM_BASE" --kfac-nesterov \
      --kfac-muon-eps "$KFAC_EPS_BASE" --kfac-muon-lr-adjustment "$KFAC_LR_ADJUSTMENT_BASE" \
      --kfac-stats-update-every "$KFAC_STATS_EVERY_BASE" --kfac-factor-update-every "$KFAC_FACTOR_EVERY_BASE" \
      --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
      --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
      --kfac-lm-rho-low 1.2 --kfac-lm-rho-high 1.8 \
      --kfac-aux-no-decay
  fi

  if contains_case "kfac_lr_low"; then
    run_train "vits16_c100_kfac_lr_low_e${EPOCHS}_s${seed}" "$seed" \
      --opt kfac_muon --lr "$KFAC_LR_LOW" \
      --kfac-damping "$KFAC_DAMPING_BASE" --kfac-momentum "$KFAC_MOMENTUM_BASE" --kfac-nesterov \
      --kfac-muon-eps "$KFAC_EPS_BASE" --kfac-muon-lr-adjustment "$KFAC_LR_ADJUSTMENT_BASE" \
      --kfac-stats-update-every "$KFAC_STATS_EVERY_BASE" --kfac-factor-update-every "$KFAC_FACTOR_EVERY_BASE" \
      --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
      --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
      --kfac-lm-rho-low 1.2 --kfac-lm-rho-high 1.8 \
      --kfac-aux-no-decay
  fi

  if contains_case "kfac_lr_high"; then
    run_train "vits16_c100_kfac_lr_high_e${EPOCHS}_s${seed}" "$seed" \
      --opt kfac_muon --lr "$KFAC_LR_HIGH" \
      --kfac-damping "$KFAC_DAMPING_BASE" --kfac-momentum "$KFAC_MOMENTUM_BASE" --kfac-nesterov \
      --kfac-muon-eps "$KFAC_EPS_BASE" --kfac-muon-lr-adjustment "$KFAC_LR_ADJUSTMENT_BASE" \
      --kfac-stats-update-every "$KFAC_STATS_EVERY_BASE" --kfac-factor-update-every "$KFAC_FACTOR_EVERY_BASE" \
      --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
      --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
      --kfac-lm-rho-low 1.2 --kfac-lm-rho-high 1.8 \
      --kfac-aux-no-decay
  fi

  if contains_case "kfac_lmoff"; then
    run_train "vits16_c100_kfac_lmoff_e${EPOCHS}_s${seed}" "$seed" \
      --opt kfac_muon --lr "$KFAC_LR_FIXED" \
      --kfac-damping "$KFAC_DAMPING_BASE" --kfac-momentum "$KFAC_MOMENTUM_BASE" --kfac-nesterov \
      --kfac-muon-eps "$KFAC_EPS_BASE" --kfac-muon-lr-adjustment "$KFAC_LR_ADJUSTMENT_BASE" \
      --kfac-stats-update-every "$KFAC_STATS_EVERY_BASE" --kfac-factor-update-every "$KFAC_FACTOR_EVERY_BASE" \
      --no-kfac-lm-adapt-damping --kfac-aux-no-decay
  fi

  if contains_case "kfac_rho_wide"; then
    run_train "vits16_c100_kfac_rho_wide_e${EPOCHS}_s${seed}" "$seed" \
      --opt kfac_muon --lr "$KFAC_LR_FIXED" \
      --kfac-damping "$KFAC_DAMPING_BASE" --kfac-momentum "$KFAC_MOMENTUM_BASE" --kfac-nesterov \
      --kfac-muon-eps "$KFAC_EPS_BASE" --kfac-muon-lr-adjustment "$KFAC_LR_ADJUSTMENT_BASE" \
      --kfac-stats-update-every "$KFAC_STATS_EVERY_BASE" --kfac-factor-update-every "$KFAC_FACTOR_EVERY_BASE" \
      --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
      --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
      --kfac-lm-rho-low 1.0 --kfac-lm-rho-high 2.0 \
      --kfac-aux-no-decay
  fi

  if contains_case "kfac_damp_5e5"; then
    run_train "vits16_c100_kfac_damp5e5_e${EPOCHS}_s${seed}" "$seed" \
      --opt kfac_muon --lr "$KFAC_LR_FIXED" \
      --kfac-damping 5e-5 --kfac-momentum "$KFAC_MOMENTUM_BASE" --kfac-nesterov \
      --kfac-muon-eps "$KFAC_EPS_BASE" --kfac-muon-lr-adjustment "$KFAC_LR_ADJUSTMENT_BASE" \
      --kfac-stats-update-every "$KFAC_STATS_EVERY_BASE" --kfac-factor-update-every "$KFAC_FACTOR_EVERY_BASE" \
      --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
      --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
      --kfac-lm-rho-low 1.2 --kfac-lm-rho-high 1.8 \
      --kfac-aux-no-decay
  fi

  if contains_case "kfac_damp_2e4"; then
    run_train "vits16_c100_kfac_damp2e4_e${EPOCHS}_s${seed}" "$seed" \
      --opt kfac_muon --lr "$KFAC_LR_FIXED" \
      --kfac-damping 2e-4 --kfac-momentum "$KFAC_MOMENTUM_BASE" --kfac-nesterov \
      --kfac-muon-eps "$KFAC_EPS_BASE" --kfac-muon-lr-adjustment "$KFAC_LR_ADJUSTMENT_BASE" \
      --kfac-stats-update-every "$KFAC_STATS_EVERY_BASE" --kfac-factor-update-every "$KFAC_FACTOR_EVERY_BASE" \
      --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
      --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
      --kfac-lm-rho-low 1.2 --kfac-lm-rho-high 1.8 \
      --kfac-aux-no-decay
  fi

  if contains_case "kfac_eps0055"; then
    run_train "vits16_c100_kfac_eps0055_e${EPOCHS}_s${seed}" "$seed" \
      --opt kfac_muon --lr "$KFAC_LR_FIXED" \
      --kfac-damping "$KFAC_DAMPING_BASE" --kfac-momentum "$KFAC_MOMENTUM_BASE" --kfac-nesterov \
      --kfac-muon-eps 0.055 --kfac-muon-lr-adjustment "$KFAC_LR_ADJUSTMENT_BASE" \
      --kfac-stats-update-every "$KFAC_STATS_EVERY_BASE" --kfac-factor-update-every "$KFAC_FACTOR_EVERY_BASE" \
      --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
      --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
      --kfac-lm-rho-low 1.2 --kfac-lm-rho-high 1.8 \
      --kfac-aux-no-decay
  fi

  if contains_case "kfac_dp005"; then
    run_train "vits16_c100_kfac_dp005_e${EPOCHS}_s${seed}" "$seed" \
      --opt kfac_muon --lr "$KFAC_LR_FIXED" --drop-path 0.05 \
      --kfac-damping "$KFAC_DAMPING_BASE" --kfac-momentum "$KFAC_MOMENTUM_BASE" --kfac-nesterov \
      --kfac-muon-eps "$KFAC_EPS_BASE" --kfac-muon-lr-adjustment "$KFAC_LR_ADJUSTMENT_BASE" \
      --kfac-stats-update-every "$KFAC_STATS_EVERY_BASE" --kfac-factor-update-every "$KFAC_FACTOR_EVERY_BASE" \
      --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
      --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
      --kfac-lm-rho-low 1.2 --kfac-lm-rho-high 1.8 \
      --kfac-aux-no-decay
  fi

  if contains_case "kfac_wd005"; then
    run_train "vits16_c100_kfac_wd005_e${EPOCHS}_s${seed}" "$seed" \
      --opt kfac_muon --lr "$KFAC_LR_FIXED" --weight-decay 0.05 \
      --kfac-damping "$KFAC_DAMPING_BASE" --kfac-momentum "$KFAC_MOMENTUM_BASE" --kfac-nesterov \
      --kfac-muon-eps "$KFAC_EPS_BASE" --kfac-muon-lr-adjustment "$KFAC_LR_ADJUSTMENT_BASE" \
      --kfac-stats-update-every "$KFAC_STATS_EVERY_BASE" --kfac-factor-update-every "$KFAC_FACTOR_EVERY_BASE" \
      --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
      --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
      --kfac-lm-rho-low 1.2 --kfac-lm-rho-high 1.8 \
      --kfac-aux-no-decay
  fi

  # Optional ResNet-prior transfer cases (not default).
  if contains_case "kfac_resnet_transfer"; then
    run_train "vits16_c100_kfac_transfer_e${EPOCHS}_s${seed}" "$seed" \
      --opt kfac_muon --lr "$MUON_LR_FIXED" \
      --warmup-epochs 5 --drop-path 0.0 \
      --kfac-damping 2e-5 --kfac-momentum 0.9 --kfac-nesterov \
      --kfac-muon-eps 0.055 --kfac-muon-lr-adjustment match_rms_adamw \
      --kfac-stats-update-every 2 --kfac-factor-update-every 2 \
      --no-kfac-lm-adapt-damping --kfac-aux-no-decay
  fi

  if contains_case "kfac_resnet_transfer_lmon"; then
    run_train "vits16_c100_kfac_transfer_lmon_e${EPOCHS}_s${seed}" "$seed" \
      --opt kfac_muon --lr "$MUON_LR_FIXED" \
      --warmup-epochs 5 --drop-path 0.0 \
      --kfac-damping 2e-5 --kfac-momentum 0.9 --kfac-nesterov \
      --kfac-muon-eps 0.055 --kfac-muon-lr-adjustment match_rms_adamw \
      --kfac-stats-update-every 2 --kfac-factor-update-every 2 \
      --kfac-lm-adapt-damping --kfac-lm-update-every 5 --kfac-lm-log-every 200 \
      --kfac-lm-decay-base 0.995 --kfac-lm-damping-min 5e-5 --kfac-lm-damping-max 1000 \
      --kfac-lm-rho-low 1.1 --kfac-lm-rho-high 1.8 \
      --kfac-aux-no-decay
  fi
done

echo
echo "Sweep complete."
