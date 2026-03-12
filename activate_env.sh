#!/usr/bin/env bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "${PROJECT_DIR}")"
ENV_PATH="${PARENT_DIR}/envs/denoisefl"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"