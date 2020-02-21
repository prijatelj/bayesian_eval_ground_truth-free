#!/bin/bash

#$ -pe smp 4
#$ -N Sim5u10sS
#$ -q debug
#$ -o logs/mcmc/softmax/sim/hmc/sampling/logs/
#$ -e logs/mcmc/softmax/sim/hmc/sampling/logs/
#$ -t 1-50

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
conda activate psych_exp_tf-1.15

python3 $BASE_PATH/experiment/research/bnn/proto_bnn_mcmc.py \
    "/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/sim/samples_10/converging/hmc_2e6sssa_5u_0burn_5lag_1e6nr_5e-4ss_3nlfs_1e-2sim_1e1s_res1e4_2020-02-10_18-15-54/data_for_bnn.json" \
    --output_dir "/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/sim/samples_10/sampling/hmc_5u_6.7e-3ss_1e2res/s2e4/job_$SGE_TASK_ID/" \
    --bnn_weights_file "/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/sim/samples_10/converging/hmc_5u_6.7e-3ss_1e5res/last_weights.json" \
    --num_hidden 5 \
    --cpu 1 \
    --cpu_cores 4 \
    --gpu 0 \
    --kernel_id 'HamiltonianMonteCarlo' \
    --num_results 100 \
    --step_size 0.0067 \
    --num_leapfrog_steps 3 \
    --lag 12840 \
    --burnin 12840 \
    --parallel_chains 4 \
    --scale_identity_multiplier 0.01 \
    --log_level 'INFO' \
    --log_file "$BASE_PATH/logs/mcmc/softmax/sim/hmc/sampling/hmc_$SGE_TASK_ID.log"

    # lag selected for < 0.1 acf
