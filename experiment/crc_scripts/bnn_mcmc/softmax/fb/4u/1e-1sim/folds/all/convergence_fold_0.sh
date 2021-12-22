#!/bin/bash

#$ -pe smp 4
#$ -N fbF0A_c
#$ -q long
#$ -o $HOME/Public/psych_metric/logs/mcmc/softmax/fb/4u_1e-1/train/logs/
#$ -e $HOME/Public/psych_metric/logs/mcmc/softmax/fb/4u_1e-1/train/logs/
#$ -t 1-10

BASE_PATH="$HOME/Public/psych_metric"
SRC_PATH="/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/fb/4u_1e-1"

# set up the environment
module add conda
conda activate psych_exp_tf-1.15

# set step_size for this job id based on a step size
STEP_SIZE=$(python3 -c "print(f'{(0.0029 - 0.0001 * ($SGE_TASK_ID - 1)):.4f}')")

python3 $BASE_PATH/experiment/research/bnn/proto_bnn_mcmc.py \
    "$HOME/scratch_21/under_over/fb_fq_16b_under_over/fb_full_trained/folds/proto_bnn_format_all_fold_0_train.json" \
    --output_dir "$SRC_PATH/folds/all/fold_0/converging/hmc_4u_1e-1sim_3nlfs_0lag_1e6res/"$STEP_SIZE"ss/2e7burn/" \
    --bnn_weights_file "$SRC_PATH/all_half_1/sssa/hmc_4u/_2020-04-30_04-00-36/last_weights.json" \
    --num_hidden 4 \
    --cpu 1 \
    --cpu_cores 4 \
    --gpu 0 \
    --kernel_id 'HamiltonianMonteCarlo' \
    --num_results 1000000 \
    --step_size  $STEP_SIZE \
    --num_leapfrog_steps 3 \
    --lag 0 \
    --burnin 20000000 \
    --scale_identity_multiplier 0.1 \
    --log_level 'INFO' \
    --log_file "$BASE_PATH/logs/mcmc/softmax/fb/4u_1e-1/train/hmc_converging.log"
