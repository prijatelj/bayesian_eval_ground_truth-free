#!/bin/bash

#$ -pe smp 4        # Specify parallel environment and legal core size
#$ -N fb_lpe_b       # Specify job name
#$ -q gpu
#$ -l gpu_card=1
#$ -o logs/fb/fq/logs/
#$ -e logs/fb/fq/logs/

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module load cudnn cuda tensorflow
conda activate metric

# get the random seed for the specific job from file.
#SEED="$(sed "$SGE_TASK_ID q;d" $BASE_PATH/experiment/random_seeds/random_seeds_count-100.txt)"
# TODO perhaps perform the experiment on all seeds as separate jobs (job array)

python3 sjd_log_prob_exp.py \
    "$BASE_PATH/psych_metric/datasets/facial_beauty/facial_beauty_data/" \
    --dataset_id 'FacialBeauty' \
    --label_src 'frequency' \
    --output_dir 'results/sjd/1114_exp_test/FacialBeauty/sjd_baselines.json' \
    --weights_file 'resnext50.h5' \
    --dir_path 'results/predictors/early_stop/fq/FacialBeauty/resnext50/1114023021/2019-11-09_00-02-42/5_fold_cv/' \
    --log_level 'INFO' \
    --processes 4
