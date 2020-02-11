#!/bin/bash

#$ -pe smp 4
#$ -N db_hmc_s
#$ -q debug
#$ -o logs/mcmc/softmax/db/hmc/logs/
#$ -e logs/mcmc/softmax/db/hmc/logs/
#$ -t 1-10

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
conda activate psych_exp_tf-1.15

python3 $BASE_PATH/experiment/research/bnn/proto_bnn_mcmc.py \
    "$BASE_PATH/results/debug/bnn_softmax/MCMC/HMC/SSSA/hmc_2e6sssa_5u_0burn_5lag_1e6nr_5e-4ss_3nlfs_1e-2sim_1e1s_res1e4/_2020-02-10_18-15-54/data_for_bnn.json" \
    --output_dir "/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/MCMC/HMC/SSSA/hmc_2e6sssa_5u_1e7burn_5lag_1e6nr_5e-4ss_5nlfs_1e-2sim_1e1s_res1e2/sampling/s8e4/job_$SGE_TASK_ID/" \
    --bnn_weights_file "$BASE_PATH/results/debug/bnn_softmax/MCMC/HMC/SSSA/hmc_2e6sssa_5u_0burn_5lag_1e6nr_5e-4ss_3nlfs_1e-2sim_1e1s_res1e4/_2020-02-10_18-15-54/cont/_2020-02-10_20-12-56/last_weights.json" \
    --num_hidden 5 \
    --cpu 1 \
    --cpu_cores 4 \
    --gpu 0 \
    --kernel_id 'HamiltonianMonteCarlo' \
    --num_results 10000 \
    --step_size 0.010185 \
    --num_leapfrog_steps 3 \
    --lag 440 \
    --burnin 440 \
    --parallel_chains 4 \
    --scale_identity_multiplier 0.01 \
    --log_level 'INFO' \
    --log_file "$BASE_PATH/logs/mcmc/softmax/db/hmc/hmc_sampling_$SGE_TASK_ID.log"
