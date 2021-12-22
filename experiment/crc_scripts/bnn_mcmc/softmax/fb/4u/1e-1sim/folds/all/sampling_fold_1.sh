#!/bin/bash

#$ -pe smp 4
#$ -N fbF1A_s
#$ -q long
#$ -o $HOME/Public/psych_metric/logs/mcmc/softmax/fb/4u_1e-1/train/logs/
#$ -e $HOME/Public/psych_metric/logs/mcmc/softmax/fb/4u_1e-1/train/logs/
#$ -t 1-5

BASE_PATH="$HOME/Public/psych_metric"
SRC_PATH="/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/fb/4u_1e-1/folds/all/fold_1"

# set up the environment
module add conda
conda activate psych_exp_tf-1.15

# set step_size for this job id based on a step size
#STEP_SIZE=$(python3 -c "print(f'{(0.0016 - 0.0001 * ($SGE_TASK_ID - 1)):.4f}')")

python3 $BASE_PATH/experiment/research/bnn/proto_bnn_mcmc.py \
    "$HOME/scratch_21/under_over/fb_fq_16b_under_over/fb_full_trained/folds/proto_bnn_format_all_fold_1_train.json" \
    --output_dir "$SRC_PATH/sampling/hmc_4u_1e-1sim_3nlfs_0lag_1e6res/0.0014ss/2e7burn/s2e4/job_$SGE_TASK_ID" \
    --bnn_weights_file "$SRC_PATH/converging/hmc_4u_1e-1sim_3nlfs_0lag_1e6res/0.0014ss/2e7burn/_2020-06-07_02-01-51/last_weights.json" \
    --num_hidden 4 \
    --cpu 1 \
    --cpu_cores 4 \
    --gpu 0 \
    --kernel_id 'HamiltonianMonteCarlo' \
    --num_results 1000 \
    --step_size  0.0014 \
    --num_leapfrog_steps 3 \
    --lag 4245 \
    --burnin 4245 \
    --parallel_chains 4 \
    --scale_identity_multiplier 0.1 \
    --log_level 'INFO' \
    --log_file "$BASE_PATH/logs/mcmc/softmax/fb/4u_1e-1/train/sampling/job_$SGE_TASK_ID.log"
