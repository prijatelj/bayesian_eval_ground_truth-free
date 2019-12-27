#!/bin/bash

#$ -pe smp 24
#$ -N fbHMC_2e5
#$ -q long
#$ -o logs/mcmc/fb/fq/1114/logs/
#$ -e logs/mcmc/fb/fq/1114/logs/

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
conda activate tf-1.15

# get the random seed for the specific job from file.
#SEED="$(sed "$SGE_TASK_ID q;d" $BASE_PATH/experiment/random_seeds/random_seeds_count-100.txt)"
# TODO perhaps perform the experiment on all seeds as separate jobs (job array)

python3 proto_bnn_mcmc.py \
    "$BASE_PATH/results/sjd/1114_exp_test/FacialBeauty/fq_train_set_human_pred_simplex_labels.json" \
    --bnn_weights_file "$BASE_PATH/results/sjd/1114_exp_test/FacialBeauty/hmc/weights_probably_converged.json" \
    --output_dir "$BASE_PATH/results/sjd/1114_exp_test/FacialBeauty/fq/bnn_samples_2e5/" \
    --cpu_cores 24 \
    --gpu 0 \
    --log_file "$BASE_PATH/logs/mcmc/fb/fq/1114/hmc_2e5.log" \
    --log_level 'INFO' \
    --kernel_id 'HamiltonianMonteCarlo' \
    --step_size 0.00006 \
    --num_leapfrog_steps 5 \
    --num_results 10000 \
    --burnin  1640 \
    --lag 1640 \
    --parallel_chains 20
