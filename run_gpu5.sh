#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
mkdir -p logs/feddenoise_v2/cifar10_noise_scan

MAX_JOBS=3
job_count=0

run_exp () {
  dataset=$1
  noise_rate=$2
  drop_rate=$3
  logfile=$4

  python main.py \
    --device_id 0 \
    --model feddenoise_v2 \
    --dataset ${dataset} \
    --parti_num 100 \
    --online_ratio 0.1 \
    --communication_epoch 100 \
    --local_epoch 10 \
    --local_batch_size 32 \
    --local_lr 0.01 \
    --partition_mode dirichlet \
    --dir_alpha 0.3 \
    --noise_mode uniform \
    --noise_rate ${noise_rate} \
    --noise_type symmetric \
    --drop_rate ${drop_rate} \
    --denoise_strategy least_sim \
    --refresh_gap 10 \
    --evaluator_schedule 8,6,4,2 \
    --score_agg weighted_mean \
    --seed 0 > ${logfile} 2>&1 &
}

submit_job () {
  run_exp "$1" "$2" "$3" "$4"
  job_count=$((job_count + 1))
  if [ ${job_count} -ge ${MAX_JOBS} ]; then
    wait
    job_count=0
  fi
}

echo "===== GPU5 CIFAR10 NOISE-SCAN START $(date) ====="

for nr in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
  dr=$(python -c "nr=float('${nr}'); print(f'{nr + 0.05:.2f}'.rstrip('0').rstrip('.'))")
  submit_job \
    fl_cifar10 \
    ${nr} \
    ${dr} \
    logs/feddenoise_v2/cifar10_noise_scan/fl_cifar10_a0.3_nr${nr}_dr${dr}_symmetric_bs32.log
done

wait
echo "===== GPU5 CIFAR10 NOISE-SCAN END $(date) ====="