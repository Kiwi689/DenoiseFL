#!/usr/bin/env bash
set -euo pipefail

# =========================
# 路径配置
# =========================
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG_DIR="${PROJECT_DIR}/run_ours_test_drchange_logs"

mkdir -p "${LOG_DIR}"

# 如果你想自动激活环境，取消下面两行注释，并改成你的环境路径
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate /root/workspace/test/kiwi/envs/denoisefl

# =========================
# 通用实验配置
# =========================
GPU_ID=4
MAX_JOBS=4

PARTI_NUM=10
ONLINE_RATIO=1.0

PARTITION_MODE="dirichlet"
ALPHAS=(0.1 0.3 0.5 0.8)

NOISE_MODE="uniform"
NOISE_RATE=0.5
NOISE_TYPE="asymmetric"

LOCAL_EPOCH=5
LOCAL_BATCH_SIZE=64
SEED=0

DATASETS=(
  "fl_cifar10"
)

# 固定 drop_rate 实验循环
DROP_RATES=(0.1 0.2 0.3 0.4 0.45 0.5 0.55 0.7)

# =========================
# Ours (FedDenoiseV3) 固定配置
# =========================
WARMUP_ROUND=20
STAGE_ROUND=19
TEACHER_SCHEDULE="4,3,2,1"

TEACHER_SELECT_STRATEGY="least_sim"
TEACHER_SIMILARITY="full_model_cosine"
TEACHER_SCORE_MODE="teacher_mean"
WARMUP_MODE="full_model"

# =========================
# 启动作业
# =========================
launch_job() {
    local dataset="$1"
    local alpha="$2"
    local drop_rate="$3"

    local job_name="feddenoise_v3_${dataset}_alpha${alpha}_dr${drop_rate}"
    local log_file="${LOG_DIR}/${job_name}.log"

    echo "[Launch] ${job_name}"
    echo "[Log] ${log_file}"

    (
        cd "${PROJECT_DIR}"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" python main.py \
            --model feddenoise_v3 \
            --dataset "${dataset}" \
            --parti_num "${PARTI_NUM}" \
            --online_ratio "${ONLINE_RATIO}" \
            --partition_mode "${PARTITION_MODE}" \
            --dir_alpha "${alpha}" \
            --noise_mode "${NOISE_MODE}" \
            --noise_rate "${NOISE_RATE}" \
            --noise_type "${NOISE_TYPE}" \
            --local_epoch "${LOCAL_EPOCH}" \
            --local_batch_size "${LOCAL_BATCH_SIZE}" \
            --device_id 0 \
            --seed "${SEED}" \
            --warmup_round "${WARMUP_ROUND}" \
            --stage_round "${STAGE_ROUND}" \
            --teacher_schedule "${TEACHER_SCHEDULE}" \
            --drop_rate "${drop_rate}" \
            --warmup_mode "${WARMUP_MODE}" \
            --teacher_select_strategy "${TEACHER_SELECT_STRATEGY}" \
            --teacher_similarity "${TEACHER_SIMILARITY}" \
            --teacher_score_mode "${TEACHER_SCORE_MODE}"
    ) > "${log_file}" 2>&1 &
}

# =========================
# 主循环
# =========================
total_jobs=0
for dataset in "${DATASETS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        for drop_rate in "${DROP_RATES[@]}"; do
            total_jobs=$((total_jobs + 1))
        done
    done
done

echo "[Info] Total jobs: ${total_jobs}"
echo "[Info] GPU: ${GPU_ID}"
echo "[Info] Max parallel jobs: ${MAX_JOBS}"
echo "[Info] Logs: ${LOG_DIR}"

for dataset in "${DATASETS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        for drop_rate in "${DROP_RATES[@]}"; do
            launch_job "${dataset}" "${alpha}" "${drop_rate}"

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
echo "[Done] All ours experiments finished."
echo "[Logs] ${LOG_DIR}"