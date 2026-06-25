#!/usr/bin/env bash
set -euo pipefail

# Create/use local venv, install CUDA 12.8 PyTorch wheels, then project deps.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install -U pip

# Force CUDA 12.8 wheels for torch/torchvision.
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# Install remaining deps from this repo.
python -m pip install -r requirements.txt

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda_runtime:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda.is_available() is False")
if torch.version.cuda is None or not torch.version.cuda.startswith("12.8"):
    raise SystemExit(f"ERROR: expected CUDA 12.8 wheels, got torch.version.cuda={torch.version.cuda!r}")
print("CUDA 12.8 environment looks good.")
PY
