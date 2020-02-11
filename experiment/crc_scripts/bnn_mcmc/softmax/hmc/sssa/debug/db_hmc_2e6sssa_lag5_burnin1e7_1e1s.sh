#!/bin/bash

#$ -pe smp 2
#$ -N db_hmc_1_sssa
#$ -q debug
#$ -o logs/mcmc/softmax/db/hmc/logs/
#$ -e logs/mcmc/softmax/db/hmc/logs/

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
conda activate psych_exp_tf-1.15

python3 experiment/research/bnn/proto_bnn_mcmc.py "__NO_this_should_not_exist" \
    --output_dir "$BASE_PATH/results/debug/bnn_softmax/MCMC/HMC/SSSA/hmc_2e6sssa_5u_1e7burn_5lag_1e6nr_5e-4ss_5nlfs_1e-2sim_1e1s_res1e2/" \
    --dim 3 \
    --num_hidden 5 \
    --num_samples 10 \
    --cpu 1 \
    --cpu_cores 2 \
    --gpu 0 \
    --kernel_id 'HamiltonianMonteCarlo' \
    --num_results 10 \
    --step_size 0.010185 \
    --num_leapfrog_steps 3 \
    --lag 5 \
    --burnin 10000000 \
    --step_adjust_id Simple \
    --num_adaptation_steps 2000000 \
    --scale_identity_multiplier 0.01 \
    --log_level 'INFO' \
    --log_file "$BASE_PATH/logs/mcmc/softmax/db/hmc/hmc_1.log"
