#!/usr/bin/env bash
set -euo pipefail

# Sync ImageNet-1k from local machine to a VAST instance.
#
# This script is intended to run on your LOCAL machine.
#
# It supports two local layouts:
#   1) Extracted folders:
#      LOCAL_IMAGENET_ROOT/train/<class>/*.JPEG
#      LOCAL_IMAGENET_ROOT/val/<class>/*.JPEG
#      -> rsyncs train/ + val/ directly.
#
#   2) Official archives:
#      LOCAL_IMAGENET_ROOT/ILSVRC2012_img_train.tar
#      LOCAL_IMAGENET_ROOT/ILSVRC2012_img_val.tar
#      LOCAL_IMAGENET_ROOT/ILSVRC2012_devkit_t12.tar.gz
#      -> uploads archives and runs remote prepare script.
#
# Usage example:
#   LOCAL_IMAGENET_ROOT="/path/to/imagenet" \
#   VAST_HOST="ssh8.vast.ai" \
#   VAST_PORT="37267" \
#   VAST_USER="root" \
#   REMOTE_REPO="/workspace/kfac-muon" \
#   REMOTE_DATA_ROOT="/workspace/data/imagenet" \
#   bash sync_imagenet1k_to_vast.sh

LOCAL_IMAGENET_ROOT="${LOCAL_IMAGENET_ROOT:-}"
VAST_HOST="${VAST_HOST:-ssh8.vast.ai}"
VAST_PORT="${VAST_PORT:-37267}"
VAST_USER="${VAST_USER:-root}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/kfac-muon}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/workspace/data/imagenet}"
RSYNC_PROGRESS="${RSYNC_PROGRESS:-1}"

if [[ -z "${LOCAL_IMAGENET_ROOT}" ]]; then
  echo "[error] Set LOCAL_IMAGENET_ROOT to your local ImageNet path."
  exit 1
fi

if [[ ! -d "${LOCAL_IMAGENET_ROOT}" ]]; then
  echo "[error] LOCAL_IMAGENET_ROOT does not exist: ${LOCAL_IMAGENET_ROOT}"
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "[error] rsync is required on local machine."
  exit 1
fi

if ! command -v ssh >/dev/null 2>&1; then
  echo "[error] ssh is required on local machine."
  exit 1
fi

TRAIN_DIR="${LOCAL_IMAGENET_ROOT}/train"
VAL_DIR="${LOCAL_IMAGENET_ROOT}/val"
TRAIN_TAR="${LOCAL_IMAGENET_ROOT}/ILSVRC2012_img_train.tar"
VAL_TAR="${LOCAL_IMAGENET_ROOT}/ILSVRC2012_img_val.tar"
DEVKIT_TAR="${LOCAL_IMAGENET_ROOT}/ILSVRC2012_devkit_t12.tar.gz"

HAVE_EXTRACTED=0
HAVE_TARS=0

if [[ -d "${TRAIN_DIR}" && -d "${VAL_DIR}" ]]; then
  HAVE_EXTRACTED=1
fi

if [[ -f "${TRAIN_TAR}" && -f "${VAL_TAR}" && -f "${DEVKIT_TAR}" ]]; then
  HAVE_TARS=1
fi

if [[ "${HAVE_EXTRACTED}" != "1" && "${HAVE_TARS}" != "1" ]]; then
  echo "[error] Could not find either:"
  echo "  - extracted train/ and val/ folders, or"
  echo "  - the 3 official tar files."
  exit 1
fi

SSH_TARGET="${VAST_USER}@${VAST_HOST}"
SSH_CMD=(ssh -p "${VAST_PORT}" "${SSH_TARGET}")
RSYNC_SSH="ssh -p ${VAST_PORT}"

echo "[check] SSH connectivity"
"${SSH_CMD[@]}" "echo connected >/dev/null"

echo "[remote] Ensure data dir exists: ${REMOTE_DATA_ROOT}"
"${SSH_CMD[@]}" "mkdir -p '${REMOTE_DATA_ROOT}'"

RSYNC_FLAGS=(-ah --partial --append-verify --stats)
if [[ "${RSYNC_PROGRESS}" == "1" ]]; then
  RSYNC_FLAGS+=(--info=progress2)
fi

if [[ "${HAVE_EXTRACTED}" == "1" ]]; then
  echo "[sync] Uploading extracted train/val folders"
  rsync "${RSYNC_FLAGS[@]}" -e "${RSYNC_SSH}" \
    "${TRAIN_DIR}/" "${SSH_TARGET}:${REMOTE_DATA_ROOT}/train/"
  rsync "${RSYNC_FLAGS[@]}" -e "${RSYNC_SSH}" \
    "${VAL_DIR}/" "${SSH_TARGET}:${REMOTE_DATA_ROOT}/val/"
else
  echo "[sync] Uploading official tar archives"
  rsync "${RSYNC_FLAGS[@]}" -e "${RSYNC_SSH}" \
    "${TRAIN_TAR}" "${VAL_TAR}" "${DEVKIT_TAR}" \
    "${SSH_TARGET}:${REMOTE_DATA_ROOT}/"

  echo "[remote] Running archive prepare script on VAST"
  "${SSH_CMD[@]}" \
    "cd '${REMOTE_REPO}' && DATA_ROOT='${REMOTE_DATA_ROOT}' PREPARE_ONLY=1 bash download_imagenet1k_vast.sh"
fi

echo "[verify] Remote dataset check"
"${SSH_CMD[@]}" "test -d '${REMOTE_DATA_ROOT}/train' && test -d '${REMOTE_DATA_ROOT}/val' && echo 'OK: train/val present'"

cat <<EOF
[done] ImageNet-1k is ready on VAST at:
  ${REMOTE_DATA_ROOT}

Train command:
  python train.py --data-dir ${REMOTE_DATA_ROOT} --dataset image_folder --train-split train --val-split val ...
EOF
