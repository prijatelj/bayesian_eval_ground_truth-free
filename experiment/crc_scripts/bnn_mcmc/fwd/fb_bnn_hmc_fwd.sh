#!/bin/bash

#$ -pe smp 24
#$ -N fb_fwd
#$ -q long
#$ -o logs/mcmc/fwd/fb/fq/1114/logs/
#$ -e logs/mcmc/fwd/fb/fq/1114/logs/

BASE_PATH="$HOME/Public/psych_metric"
cd $BASE_PATH

# set up the environment
module add conda
conda activate tf-1.15

# get the random seed for the specific job from file.
#SEED="$(sed "$SGE_TASK_ID q;d" $BASE_PATH/experiment/random_seeds/random_seeds_count-100.txt)"
# TODO perhaps perform the experiment on all seeds as separate jobs (job array)

python3 proto_bnn_mcmc.py \
    "results/sjd/1114_exp_test/FacialBeauty/fq_train_set_human_pred_simplex_labels.json" \
    --bnn_weights_file "results/sjd/1114_exp_test/FacialBeauty/fq/bnn_samples_2e2/" \
    --output_dir "results/sjd/1114_exp_test/FacialBeauty/fq/fwd/hmc_bnn_log_prob.json" \
    --num_layers 1 \
    --cpu_cores 24 \
    --gpu 0 \
    --log_file "logs/mcmc/fwd/fb/fq/1114/hmc_bnn_log_prob.log" \
    --log_level 'INFO' \
