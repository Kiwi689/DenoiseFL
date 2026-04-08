#!/usr/bin/env bash
set -euo pipefail

# =========================
# 路径配置
# =========================
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/baseline_logs_3"
mkdir -p "${LOG_DIR}"

# 如果你想自动激活环境，取消下面两行注释，并改成你的环境路径
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate /root/workspace/test/kiwi/envs/denoisefl

# =========================
# 通用实验配置
# =========================
GPU_ID=0
MAX_JOBS=3

PARTI_NUM=10
ONLINE_RATIO=1.0
COMM_EPOCH=100
PARTITION_MODE="dirichlet"
ALPHAS=(0.1 0.3 0.5 0.8)

NOISE_MODE="uniform"
NOISE_RATE=0.3
NOISE_TYPE="symmetric"

LOCAL_EPOCH=1
LOCAL_BATCH_SIZE=64
SEED=0

MODELS=(
  "fedcda"
  "fedrdn"
  "fedgloss"
)

DATASETS=(
  "fl_cifar10"
  "fl_cifar100"
  "fl_mnist"
  "fl_svhn"
)

# =========================
# 方法专属参数
# =========================
build_extra_args() {
    local model="$1"
    local dataset="$2"

    case "${model}" in
        fedavg)
            echo ""
            ;;
        fedprox)
            case "${dataset}" in
                fl_cifar100) echo "--mu 0.001" ;;
                *)           echo "--mu 0.01" ;;
            esac
            ;;
        fedcda)
            echo "--cda_history_size 3 --cda_batch_num 3 --cda_warmup_round 30 --cda_L 1.0"
            ;;
        fedrdn)
            echo "--rdn_std 0.01 --rdn_eps 1e-6"
            ;;
        fedgloss)
            echo "--rho 0.05 --beta 1.0 --server_lr 1.0"
            ;;
        *)
            echo ""
            ;;
    esac
}

# =========================
# 启动作业
# =========================
launch_job() {
    local model="$1"
    local dataset="$2"
    local alpha="$3"

    local extra_args
    extra_args="$(build_extra_args "${model}" "${dataset}")"

    local job_name="${model}_${dataset}_alpha${alpha}"
    local log_file="${LOG_DIR}/${job_name}.log"

    echo "[Launch] ${job_name}"
    echo "[Log] ${log_file}"

    (
        cd "${PROJECT_DIR}"

        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python main.py \
            --model "${model}" \
            --dataset "${dataset}" \
            --parti_num "${PARTI_NUM}" \
            --online_ratio "${ONLINE_RATIO}" \
            --communication_epoch "${COMM_EPOCH}" \
            --partition_mode "${PARTITION_MODE}" \
            --dir_alpha "${alpha}" \
            --noise_mode "${NOISE_MODE}" \
            --noise_rate "${NOISE_RATE}" \
            --noise_type "${NOISE_TYPE}" \
            --local_epoch "${LOCAL_EPOCH}" \
            --local_batch_size "${LOCAL_BATCH_SIZE}" \
            --device_id 0 \
            --seed "${SEED}" \
            ${extra_args}
    ) > "${log_file}" 2>&1 &
}

# =========================
# 主循环：最多并行 3 个
# =========================
job_count=0
total_jobs=0

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            total_jobs=$((total_jobs + 1))
        done
    done
done

echo "[Info] Total jobs: ${total_jobs}"
echo "[Info] GPU: ${GPU_ID}"
echo "[Info] Max parallel jobs: ${MAX_JOBS}"
echo "[Info] Logs: ${LOG_DIR}"

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            launch_job "${model}" "${dataset}" "${alpha}"
            job_count=$((job_count + 1))

            # 控制最大并行数
            while true; do
                running_jobs=$(jobs -pr | wc -l | tr -d ' ')
                if [[ "${running_jobs}" -lt "${MAX_JOBS}" ]]; then
                    break
                fi
                sleep 5
            done
        done
    done
done

echo "[Info] All jobs submitted. Waiting..."
wait
echo "[Done] All baseline experiments finished."
echo "[Logs] ${LOG_DIR}"