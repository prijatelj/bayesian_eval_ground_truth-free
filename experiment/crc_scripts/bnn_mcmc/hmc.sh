#!/bin/bash

#$ -pe smp 16        # Specify parallel environment and legal core size
#$ -N hmc       # Specify job name
#$ -q gpu
#$ -l gpu_card=1
#$ -o logs/mcmc/hmc/logs/
#$ -e logs/mcmc/hmc/logs/

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module conda
source activate "$HOME/TF-1.15"

python3 proto_bnn_mcmc.py
    'MCMC/HMC/hmc_10s_10u_1e4a_1e6nr_5e-4ss_5nlfs/' \
    --num_hidden 10 \
    --num_samples 10 \
    --adam_epochs 10000 \
    --cpu 1 \
    --cpu_cores 16 \
    --gpu 1 \
    --kernel_id 'HamiltonianMonteCarlo' \
    --num_results 1000000 \
    --step_size 0.0005
    --num_leapfrog_steps 5
    --log_level 'INFO'
