#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
mkdir -p logs/feddenoise_v2/cifar10

MAX_JOBS=2
job_count=0

run_exp () {
  dataset=$1
  dir_alpha=$2
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
    --dir_alpha ${dir_alpha} \
    --noise_mode uniform \
    --noise_rate 0.3 \
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

echo "===== GPU0 CIFAR10 START $(date) ====="

for a in 0.1 0.3 0.5 0.8; do
  for dr in 0.3 0.35 0.4 0.45; do
    submit_job \
      fl_cifar10 \
      ${a} \
      ${dr} \
      logs/feddenoise_v2/cifar10/fl_cifar10_a${a}_dr${dr}_nr0.3_symmetric_bs32.log
  done
done

wait
echo "===== GPU0 CIFAR10 END $(date) ====="