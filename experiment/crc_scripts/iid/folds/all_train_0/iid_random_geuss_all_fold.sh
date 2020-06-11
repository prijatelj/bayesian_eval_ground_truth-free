#!/bin/bash

#$ -pe smp 5
#$ -N fbAh_RG
#$ -q debug
#$ -o $HOME/Public/psych_metric/logs/fb/train_half_1/exp1/random_guess/logs/
#$ -e $HOME/Public/psych_metric/logs/fb/train_half_1/exp1/random_guess/logs/

export BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
conda activate psych_exp_tf-1.15

# get the random seed for the specific job from file.
#SEED="$(sed "$SGE_TASK_ID q;d" $BASE_PATH/experiment/random_seeds/random_seeds_count-100.txt)"
# TODO perhaps perform the experiment on train seeds as separate jobs (job array)

python3 "$BASE_PATH/experiment/research/measure/sjd_euclid_dist.py" \
    "$HOME/scratch_21/under_over/fb_fq_16b_under_over/fb_full_trained/folds/proto_bnn_format_all_fold_0_train.json" \
    --test_datapath "$HOME/scratch_21/under_over/fb_fq_16b_under_over/fb_full_trained/proto_bnn_format_all_fold_0_test.json" \
    --output_dir "$HOME/scratch_21/results/sjd/3965_exp_test/fb/fq/folds/all_fold_0/exp1/baselines_samples_1.45e4/" \
    --sample_size 14500 \
    --src_candidates 'iid_uniform_dirs' 'iid_dirs_adam' \
    --normalize \
    --save_conds \
    --cpu_cores 4 \
    --gpu 0 \
    --log_file "$BASE_PATH/logs/fb/train_half_1/exp1/random_guess/l2.log" \
    --log_level INFO

