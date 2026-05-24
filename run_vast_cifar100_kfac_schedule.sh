#!/usr/bin/env bash
set -euo pipefail

# KFAC-focused staged schedule wrapper for CIFAR-100 ViT-S/16.
#
# This wraps run_vast_cifar100_vits16_sweep.sh with curated case sets.
#
# Stages:
#   - rank75 (default): single-seed 75-epoch ranking sweep
#   - confirm200: multi-seed 200-epoch confirmation on chosen top configs
#   - diagnostics75: optional follow-up diagnostics (inverse factors + slower updates)
#
# Examples:
#   STAGE=rank75 PREP_DATA=1 bash run_vast_cifar100_kfac_schedule.sh
#   STAGE=confirm200 TOP_CASES="kfac_rho_wide,kfac_vit_base" SEEDS="11 12 13" PREP_DATA=0 bash run_vast_cifar100_kfac_schedule.sh
#   STAGE=diagnostics75 TARGET_CASE="kfac_rho_wide" PREP_DATA=0 bash run_vast_cifar100_kfac_schedule.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$SCRIPT_DIR}"
cd "$WORKDIR"

BASE_SCRIPT="${BASE_SCRIPT:-$SCRIPT_DIR/run_vast_cifar100_vits16_sweep.sh}"
if [[ ! -x "$BASE_SCRIPT" ]]; then
  echo "Base sweep script not found or not executable: $BASE_SCRIPT"
  exit 1
fi

STAGE="${STAGE:-rank75}"              # rank75 | confirm200 | diagnostics75
PREP_DATA="${PREP_DATA:-0}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/logs/timm_train}"

# You can override seeds globally.
SEEDS="${SEEDS:-}"

run_base() {
  local phase="$1"
  local cases="$2"
  local seeds="$3"
  PHASE="$phase" \
  CASES="$cases" \
  SEEDS="$seeds" \
  PREP_DATA="$PREP_DATA" \
  DRY_RUN="$DRY_RUN" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  bash "$BASE_SCRIPT"
}

if [[ "$STAGE" == "rank75" ]]; then
  # Focused KFAC ranking set (no muon runs).
  RANK_CASES="${RANK_CASES:-kfac_vit_base,kfac_rho_wide,kfac_wd005,kfac_dp005,kfac_eps0055,kfac_lr_high,kfac_lr_low,kfac_lmoff,kfac_damp_5e5,kfac_damp_2e4}"
  RANK_SEEDS="${SEEDS:-12}"
  echo "==> Stage: rank75"
  echo "==> Cases: $RANK_CASES"
  echo "==> Seeds: $RANK_SEEDS"
  run_base "pilot" "$RANK_CASES" "$RANK_SEEDS"
  exit 0
fi

if [[ "$STAGE" == "confirm200" ]]; then
  if [[ -z "${TOP_CASES:-}" ]]; then
    echo "TOP_CASES is required for STAGE=confirm200 (e.g., TOP_CASES=\"kfac_rho_wide,kfac_vit_base\")"
    exit 1
  fi
  CONFIRM_SEEDS="${SEEDS:-11 12 13}"
  echo "==> Stage: confirm200"
  echo "==> Cases: $TOP_CASES"
  echo "==> Seeds: $CONFIRM_SEEDS"
  run_base "full" "$TOP_CASES" "$CONFIRM_SEEDS"
  exit 0
fi

if [[ "$STAGE" == "diagnostics75" ]]; then
  if [[ -z "${TARGET_CASE:-}" ]]; then
    echo "TARGET_CASE is required for STAGE=diagnostics75 (e.g., TARGET_CASE=\"kfac_rho_wide\")"
    exit 1
  fi
  DIAG_SEEDS="${SEEDS:-12}"
  echo "==> Stage: diagnostics75"
  echo "==> Target case family: $TARGET_CASE"
  echo "==> Seeds: $DIAG_SEEDS"
  echo
  echo "This stage runs two manual commands per seed:"
  echo "  1) +inverse factors"
  echo "  2) slower KFAC updates (4/4)"
  echo
  for seed in $DIAG_SEEDS; do
    exp_base="vits16_c100_${TARGET_CASE}_diag_e75_s${seed}"
    cmd_inv=(
      python3 train.py
      --data-dir /dev/shm/cifar100
      --dataset image_folder
      --train-split train
      --val-split val
      --num-classes 100
      --model vit_small_patch16_224
      --epochs 75
      --batch-size 128
      --validation-batch-size 128
      --workers 8
      --weight-decay 0.07
      --sched cosine
      --warmup-epochs 8
      --min-lr 1e-5
      --mixup 0.2
      --cutmix 0.2
      --mixup-off-epoch 56
      --smoothing 0.1
      --reprob 0.1
      --drop-path 0.1
      --opt-betas 0.9 0.95
      --seed "$seed"
      --log-interval 200
      --val-interval 1
      --checkpoint-hist 2
      --output "$OUTPUT_ROOT"
      --experiment "${exp_base}_inv"
      --amp
      --amp-dtype bfloat16
      --opt kfac_muon
      --lr 5.5e-4
      --kfac-damping 1.5e-4
      --kfac-momentum 0.9
      --kfac-nesterov
      --kfac-muon-eps 0.038
      --kfac-muon-lr-adjustment match_rms_adamw
      --kfac-stats-update-every 2
      --kfac-factor-update-every 2
      --kfac-lm-adapt-damping
      --kfac-lm-update-every 5
      --kfac-lm-log-every 200
      --kfac-lm-decay-base 0.995
      --kfac-lm-damping-min 5e-5
      --kfac-lm-damping-max 1000
      --kfac-lm-rho-low 1.2
      --kfac-lm-rho-high 1.8
      --kfac-aux-no-decay
      --kfac-use-inverse-factors
    )
    cmd_u44=(
      python3 train.py
      --data-dir /dev/shm/cifar100
      --dataset image_folder
      --train-split train
      --val-split val
      --num-classes 100
      --model vit_small_patch16_224
      --epochs 75
      --batch-size 128
      --validation-batch-size 128
      --workers 8
      --weight-decay 0.07
      --sched cosine
      --warmup-epochs 8
      --min-lr 1e-5
      --mixup 0.2
      --cutmix 0.2
      --mixup-off-epoch 56
      --smoothing 0.1
      --reprob 0.1
      --drop-path 0.1
      --opt-betas 0.9 0.95
      --seed "$seed"
      --log-interval 200
      --val-interval 1
      --checkpoint-hist 2
      --output "$OUTPUT_ROOT"
      --experiment "${exp_base}_u44"
      --amp
      --amp-dtype bfloat16
      --opt kfac_muon
      --lr 5.5e-4
      --kfac-damping 1.5e-4
      --kfac-momentum 0.9
      --kfac-nesterov
      --kfac-muon-eps 0.038
      --kfac-muon-lr-adjustment match_rms_adamw
      --kfac-stats-update-every 4
      --kfac-factor-update-every 4
      --kfac-lm-adapt-damping
      --kfac-lm-update-every 5
      --kfac-lm-log-every 200
      --kfac-lm-decay-base 0.995
      --kfac-lm-damping-min 5e-5
      --kfac-lm-damping-max 1000
      --kfac-lm-rho-low 1.2
      --kfac-lm-rho-high 1.8
      --kfac-aux-no-decay
    )

    echo "==> Running diagnostics for seed=$seed"
    printf ' %q' "${cmd_inv[@]}"
    echo
    if [[ "$DRY_RUN" != "1" ]]; then
      "${cmd_inv[@]}"
    fi
    printf ' %q' "${cmd_u44[@]}"
    echo
    if [[ "$DRY_RUN" != "1" ]]; then
      "${cmd_u44[@]}"
    fi
  done
  exit 0
fi

echo "Unsupported STAGE: $STAGE (expected rank75, confirm200, diagnostics75)"
exit 1
