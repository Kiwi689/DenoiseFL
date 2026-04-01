#!/usr/bin/env bash
set -e

PYTHON_VERSION="3.10"

# 当前脚本所在目录 = 项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 项目的父目录，例如 .../kiwi
PARENT_DIR="$(dirname "${PROJECT_DIR}")"

# 环境放在与 DenoiseFL 平级的 envs/denoisefl
ENV_PATH="${PARENT_DIR}/envs/denoisefl"

echo "[Step 1] Project dir: ${PROJECT_DIR}"
echo "[Step 2] Env path: ${ENV_PATH}"

echo "[Step 3] Checking conda..."
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found."
    echo "Please install Miniconda or load conda first."
    exit 1
fi

echo "[Step 4] Loading conda..."
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[Step 5] Creating environment..."
if [ -d "${ENV_PATH}" ]; then
    echo "Environment already exists at ${ENV_PATH}. Skip creation."
else
    mkdir -p "$(dirname "${ENV_PATH}")"
    conda create -p "${ENV_PATH}" python="${PYTHON_VERSION}" -y
fi

echo "[Step 6] Activating environment..."
conda activate "${ENV_PATH}"

echo "[Step 7] Upgrading pip..."
python -m pip install --upgrade pip

echo "[Step 8] Installing requirements..."
pip install -r "${PROJECT_DIR}/requirements.txt"

echo "[Step 9] Checking torch..."
python - <<'PY'
try:
    import torch
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("Torch check failed:", e)
PY

echo "[Done] Environment setup finished."
echo "Use this command to activate it next time:"
echo "source activate_env.sh"