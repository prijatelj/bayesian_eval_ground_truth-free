#!/bin/bash

#$ -pe smp 16
#$ -N nuts_1_da
#$ -q long
#$ -o logs/mcmc/nuts/logs/
#$ -e logs/mcmc/nuts/logs/

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
source activate "$HOME/TF-1.15"

python3 proto_bnn_mcmc.py "$BASE_PATH/MCMC/NUTS/dual_avg/nuts_10u_1e5nr_1e4burn_6e3da_5lag_5e-4ss_1e1s/" \
    --num_samples 10 \
    --cpu 1 \
    --cpu_cores 16 \
    --gpu 0 \
    --num_hidden 10 \
    --kernel_id 'NoUTurnSampler' \
    --num_results 100000 \
    --step_size 0.0005 \
    --lag 5 \
    --burnin 10000 \
    --num_adaptation_steps 6000 \
    --step_adjust_id 'DualAveraging' \
    --log_level 'INFO' \
    --log_file "$BASE_PATH/logs/mcmc/nuts/nuts_1_da.log"
