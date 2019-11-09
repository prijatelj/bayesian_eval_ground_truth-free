#!/bin/bash

#$ -pe smp 4        # Specify parallel environment and legal core size
#$ -N fb_fq       # Specify job name
#$ -q gpu
#$ -l gpu_card=1
#$ -o logs/fb/fq/logs/
#$ -e logs/fb/fq/logs/
#$ -t 1-25

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module load cudnn cuda tensorflow
conda activate metric

# get the random seed for the specific job from file.
SEED="$(sed "$SGE_TASK_ID q;d" $BASE_PATH/experiment/random_seeds/random_seeds_count-100.txt)"

python3 "$BASE_PATH/predictors.py" \
    "$BASE_PATH/psych_metric/datasets/facial_beauty/facial_beauty_data/" \
    --dataset_id FacialBeauty \
    --cpu 1 \
    --cpu_cores 4 \
    --gpu 1 \
    --log_level INFO \
    --log_file 'logs/fb/fq/$SEED-resnext50.log' \
    --output_dir "$BASE_PATH/results/predictors/early_stop/fq/" \
    --random_seeds "$SEED" \
    --epochs 200 \
    --batch_size 16 \
    --model_id resnext50 \
    --label_src frequency \
    --patience 5 \
    --restore_best_weights
