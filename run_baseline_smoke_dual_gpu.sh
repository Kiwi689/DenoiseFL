#!/usr/bin/env bash
set -euo pipefail

# =========================
# 基本配置
# =========================
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PATH="$(dirname "${PROJECT_DIR}")/envs/denoisefl"

# 使用的物理 GPU
GPUS=(6 7)

# 每张卡同时跑几个任务
# 建议先 2；如果你确认显存很富裕，可以改成 3
MAX_PER_GPU=3

# 通用实验配置（最小化 smoke test）
PARTI_NUM=10
ONLINE_RATIO=1.0
COMM_EPOCH=4
PARTITION_MODE="dirichlet"
DIR_ALPHA=0.3
NOISE_MODE="uniform"
NOISE_RATE=0.3
NOISE_TYPE="symmetric"
LOCAL_EPOCH=1
LOCAL_BATCH_SIZE=64
SEED=0

LOG_DIR="${PROJECT_DIR}/smoke_logs"
mkdir -p "${LOG_DIR}"

# =========================
# 激活环境
# =========================
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

echo "[Info] Project dir: ${PROJECT_DIR}"
echo "[Info] Env path   : ${ENV_PATH}"
echo "[Info] Log dir    : ${LOG_DIR}"
echo "[Info] GPUs       : ${GPUS[*]}"
echo "[Info] Max/GPU    : ${MAX_PER_GPU}"

python - <<'PY'
try:
    import torch
    print("[Torch] version:", torch.__version__)
    print("[Torch] cuda available:", torch.cuda.is_available())
    print("[Torch] device count:", torch.cuda.device_count())
except Exception as e:
    print("[Torch] check failed:", e)
PY

# =========================
# 任务定义
# 每项格式：
# "model|dataset|extra_args"
# =========================
TASKS=(
  "fedavg|fl_cifar10|"
  "fedavg|fl_cifar100|"
  "fedavg|fl_mnist|"
  "fedavg|fl_svhn|"

  "fedprox|fl_cifar10|--mu 0.01"
  "fedprox|fl_cifar100|--mu 0.001"
  "fedprox|fl_mnist|--mu 0.01"
  "fedprox|fl_svhn|--mu 0.01"

  "fedcda|fl_cifar10|--cda_history_size 3 --cda_batch_num 3 --cda_warmup_round 2 --cda_L 1.0"
  "fedcda|fl_cifar100|--cda_history_size 3 --cda_batch_num 3 --cda_warmup_round 2 --cda_L 1.0"
  "fedcda|fl_mnist|--cda_history_size 3 --cda_batch_num 3 --cda_warmup_round 2 --cda_L 1.0"
  "fedcda|fl_svhn|--cda_history_size 3 --cda_batch_num 3 --cda_warmup_round 2 --cda_L 1.0"

  "fedrdn|fl_cifar10|--rdn_std 0.01 --rdn_eps 1e-6"
  "fedrdn|fl_cifar100|--rdn_std 0.01 --rdn_eps 1e-6"
  "fedrdn|fl_mnist|--rdn_std 0.01 --rdn_eps 1e-6"
  "fedrdn|fl_svhn|--rdn_std 0.01 --rdn_eps 1e-6"

  "fedgloss|fl_cifar10|--rho 0.05 --beta 1.0 --server_lr 1.0"
  "fedgloss|fl_cifar100|--rho 0.05 --beta 1.0 --server_lr 1.0"
  "fedgloss|fl_mnist|--rho 0.05 --beta 1.0 --server_lr 1.0"
  "fedgloss|fl_svhn|--rho 0.05 --beta 1.0 --server_lr 1.0"
)

# =========================
# GPU 上的 PID 池
# =========================
declare -a GPU6_PIDS=()
declare -a GPU7_PIDS=()

get_pid_array_name() {
    local gpu="$1"
    if [[ "$gpu" == "6" ]]; then
        echo "GPU6_PIDS"
    elif [[ "$gpu" == "7" ]]; then
        echo "GPU7_PIDS"
    else
        echo "ERROR_UNKNOWN_GPU"
    fi
}

cleanup_finished_pids() {
    local gpu="$1"
    local arr_name
    arr_name="$(get_pid_array_name "$gpu")"

    local -n arr_ref="$arr_name"
    local new_arr=()

    for pid in "${arr_ref[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            new_arr+=("$pid")
        fi
    done

    arr_ref=("${new_arr[@]}")
}

wait_for_slot() {
    local gpu="$1"
    local arr_name
    arr_name="$(get_pid_array_name "$gpu")"
    local -n arr_ref="$arr_name"

    while true; do
        cleanup_finished_pids "$gpu"
        if [[ "${#arr_ref[@]}" -lt "${MAX_PER_GPU}" ]]; then
            break
        fi
        sleep 3
    done
}

launch_one() {
    local gpu="$1"
    local model="$2"
    local dataset="$3"
    local extra_args="$4"

    local arr_name
    arr_name="$(get_pid_array_name "$gpu")"
    local -n arr_ref="$arr_name"

    local log_file="${LOG_DIR}/${model}_${dataset}_gpu${gpu}.log"

    echo "[Launch] GPU=${gpu} | model=${model} | dataset=${dataset}"
    echo "[Log] ${log_file}"

    (
        cd "${PROJECT_DIR}"
        CUDA_VISIBLE_DEVICES="${gpu}" python main.py \
            --model "${model}" \
            --dataset "${dataset}" \
            --parti_num "${PARTI_NUM}" \
            --online_ratio "${ONLINE_RATIO}" \
            --communication_epoch "${COMM_EPOCH}" \
            --partition_mode "${PARTITION_MODE}" \
            --dir_alpha "${DIR_ALPHA}" \
            --noise_mode "${NOISE_MODE}" \
            --noise_rate "${NOISE_RATE}" \
            --noise_type "${NOISE_TYPE}" \
            --local_epoch "${LOCAL_EPOCH}" \
            --local_batch_size "${LOCAL_BATCH_SIZE}" \
            --device_id 0 \
            --seed "${SEED}" \
            ${extra_args}
    ) > "${log_file}" 2>&1 &

    local pid=$!
    arr_ref+=("$pid")
    echo "[PID] ${pid} -> GPU ${gpu}"
}

# =========================
# 分发任务
# 轮流发到 GPU 6 / 7
# =========================
gpu_index=0

for task in "${TASKS[@]}"; do
    IFS='|' read -r model dataset extra_args <<< "${task}"

    gpu="${GPUS[$gpu_index]}"
    wait_for_slot "$gpu"
    launch_one "$gpu" "$model" "$dataset" "$extra_args"

    gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))
done

# =========================
# 等待所有任务结束
# =========================
echo "[Info] All tasks submitted. Waiting for completion..."

while true; do
    cleanup_finished_pids 6
    cleanup_finished_pids 7

    total_left=$(( ${#GPU6_PIDS[@]} + ${#GPU7_PIDS[@]} ))
    if [[ "$total_left" -eq 0 ]]; then
        break
    fi

    echo "[Info] Remaining jobs: ${total_left} (GPU6=${#GPU6_PIDS[@]}, GPU7=${#GPU7_PIDS[@]})"
    sleep 5
done

echo "[Done] All smoke-test jobs finished."
echo "[Logs] Check ${LOG_DIR}"