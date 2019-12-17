#!/bin/bash

#$ -pe smp 16        # Specify parallel environment and legal core size
#$ -N hmc_1_sssa       # Specify job name
#$ -q long
#$ -o logs/mcmc/hmc/logs/
#$ -e logs/mcmc/hmc/logs/

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
source activate "$HOME/TF-1.15"

python3 proto_bnn_mcmc.py "$BASE_PATH/MCMC/HMC/SSSA/hmc_6e6sssa_10u_1e7burn_5lag_1e6nr_5e-4ss_5nlfs_1e1s/" \
    --num_hidden 10 \
    --num_samples 10 \
    --cpu 1 \
    --cpu_cores 16 \
    --gpu 0 \
    --kernel_id 'HamiltonianMonteCarlo' \
    --num_results 1000000 \
    --step_size 0.0005 \
    --num_leapfrog_steps 5 \
    --lag 5 \
    --burnin 10000000 \
    --num_adaptation_steps 6000000 \
    --log_level 'INFO'
