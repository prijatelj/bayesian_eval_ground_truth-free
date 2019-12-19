#!/bin/bash

#$ -pe smp 16
#$ -N lm_lpe_dir_mean
#$ -q long
#$ -o logs/fb/fq/lpe/logs/
#$ -e logs/fb/fq/lpe/logs/

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
conda activate tf-1.15

# get the random seed for the specific job from file.
#SEED="$(sed "$SGE_TASK_ID q;d" $BASE_PATH/experiment/random_seeds/random_seeds_count-100.txt)"
# TODO perhaps perform the experiment on all seeds as separate jobs (job array)

python3 sjd_log_prob_exp.py \
    "$BASE_PATH/psych_metric/datasets/crowd_layer/crowd_layer_data/" \
    --dataset_id 'LabelMe' \
    --label_src 'frequency' \
    --output_dir "$BASE_PATH/results/sjd/1114_exp_test/LabelMe/fq/sjd_dir-mean_mvn-umvue.json" \
    --weights_file 'vgg16.h5' \
    --dir_path "$BASE_PATH/results/predictors/early_stop/fq/LabelMe/vgg16/1114023021/2019-11-08_20-09-30/5_fold_cv/" \
    --cpu_cores 16 \
    --log_file "$BASE_PATH/logs/log_prob_exp/dir-mean_mvn-umvue.log" \
    --log_level 'INFO' \
    --src_candidates 'dir-mean_mvn-umvu'
