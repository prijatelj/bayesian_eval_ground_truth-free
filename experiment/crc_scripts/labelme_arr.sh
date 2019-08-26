#!/bin/bash

#$ -pe smp 4                # Specify parallel environment and legal core size
#$ -N labv_mv		# Specify job name
#$ -q gpu@@cvrl_gpu
#$ -l gpu_card=1
#$ -o logs/labelme/mv/logs/
#$ -e logs/labelme/mv/logs/
#$ -t 1-30

BASE_PATH="$HOME/Public/psych_metric"

module load cudnn cuda tensorflow

seed="$(sed $SGE_TASK_IDq;d $BASE_PATH/experiment/random_seeds/random_seeds_count-30.txt)"

python3 "$BASE_PATH/psych_metric/predictors.py" \
    "$BASE_PATH/psych_metric/datasets/crowd_layer/" \
    --cpu 1 \
    --cpu_cores 4 \
    --gpu 1 \
    --log_level INFO \
    --log_file 'logs/label_me/vgg16_mv.log' \
    --output_dir "$BASE_PATH/results/predictors/" \
    --random_seeds "$seed"
