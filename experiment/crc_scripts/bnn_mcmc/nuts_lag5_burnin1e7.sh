#!/bin/bash

#$ -pe smp 16        # Specify parallel environment and legal core size
#$ -N nuts       # Specify job name
#$ -q gpu
#$ -l gpu_card=1
#$ -o logs/mcmc/nuts/logs/
#$ -e logs/mcmc/nuts/logs/

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
source activate "$HOME/TF-1.15"

python3 proto_bnn_mcmc.py "$BASE_PATH/MCMC/NUTS/nuts_10s_10u_1e4a_1e6nr_1e7burn_5lag_5e-4ss/" \
    --num_hidden 10 \
    --num_samples 10 \
    --adam_epochs 10000 \
    --cpu 1 \
    --cpu_cores 16 \
    --gpu 1 \
    --kernel_id 'NoUTurnSampler' \
    --num_results 1000000 \
    --step_size 0.0005 \
    --lag 5 \
    --burnin 10000000 \
    --log_level 'INFO' \
