#!/bin/bash

#$ -pe smp 4
#$ -N cHh_s
#$ -q long
#$ -o $HOME/Public/psych_metric/logs/mcmc/softmax/cha/19u_1e-1sim/all_half_1/half/sampling/logs/
#$ -e $HOME/Public/psych_metric/logs/mcmc/softmax/cha/19u_1e-1sim/all_half_1/half/sampling/logs/
#$ -t 1-5

BASE_PATH="$HOME/Public/psych_metric"
SRC_PATH="/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/cha/19u_1e-1sim/all_half_1/half"

# set up the environment
module add conda
conda activate psych_exp_tf-1.15

python3 $BASE_PATH/experiment/research/bnn/proto_bnn_mcmc.py \
    "$HOME/scratch_21/under_over/cha_fq_under_over/cha_half/ckpt1_proto_bnn_format_all_half_1.json" \
    --output_dir "$SRC_PATH/sampling/long_hmc_19u_0lag_1e6res_2e-3ss_1e7burn/s2e4/job_$SGE_TASK_ID" \
    --bnn_weights_file "$SRC_PATH/converging/long_hmc_19u_0lag_1e6res/2e-3ss/1e7burn/last_weights.json" \
    --num_hidden 19 \
    --num_layers 1 \
    --cpu 1 \
    --cpu_cores 4 \
    --gpu 0 \
    --kernel_id 'HamiltonianMonteCarlo' \
    --num_results 800 \
    --step_size  0.002 \
    --num_leapfrog_steps 3 \
    --lag 4349 \
    --burnin 4349 \
    --parallel_chains 4 \
    --scale_identity_multiplier 0.1 \
    --log_level 'INFO' \
    --log_file "$BASE_PATH/logs/mcmc/softmax/cha/19u_1e-1sim/all_half_1/half/sampling/job_$SGE_TASK_ID.log"
