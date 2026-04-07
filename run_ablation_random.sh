#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/ablation_logs_random"
mkdir -p "${LOG_DIR}"

GPU_ID=7
MAX_JOBS=3

PARTI_NUM=10
ONLINE_RATIO=1.0
PARTITION_MODE="dirichlet"
ALPHAS=(0.1 0.3 0.5 0.8)

NOISE_MODE="uniform"
NOISE_RATE=0.3
NOISE_TYPE="symmetric"

LOCAL_EPOCH=10
LOCAL_BATCH_SIZE=64
SEED=0

DATASETS=(
  "fl_cifar10"
  "fl_cifar100"
  "fl_mnist"
  "fl_svhn"
)

WARMUP_ROUND=10
STAGE_ROUND=50
TEACHER_SCHEDULE="4,3,2,1"
DROP_RATE=0.15

TEACHER_SELECT_STRATEGY="random"
TEACHER_SIMILARITY="backbone_cosine"
TEACHER_SCORE_MODE="teacher_mean"
WARMUP_MODE="backbone_only"

launch_job() {
    local dataset="$1"
    local alpha="$2"

    local job_name="feddenoise_v3_${dataset}_alpha${alpha}_random"
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
            --drop_rate "${DROP_RATE}" \
            --warmup_mode "${WARMUP_MODE}" \
            --teacher_select_strategy "${TEACHER_SELECT_STRATEGY}" \
            --teacher_similarity "${TEACHER_SIMILARITY}" \
            --teacher_score_mode "${TEACHER_SCORE_MODE}"
    ) > "${log_file}" 2>&1 &
}

echo "[Info] GPU=${GPU_ID} | Max parallel=${MAX_JOBS} | Logs=${LOG_DIR}"

for dataset in "${DATASETS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        launch_job "${dataset}" "${alpha}"

        while true; do
            running_jobs=$(jobs -pr | wc -l | tr -d ' ')
            if [[ "${running_jobs}" -lt "${MAX_JOBS}" ]]; then
                break
            fi
            sleep 5
        done
    done
done

echo "[Info] All jobs submitted. Waiting..."
wait
echo "[Done] random ablation finished."