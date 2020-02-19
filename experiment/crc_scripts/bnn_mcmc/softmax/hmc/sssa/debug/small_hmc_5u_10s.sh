#!/bin/bash

#$ -pe smp 2
#$ -N Sim5u10s
#$ -q debug
#$ -o logs/mcmc/softmax/sim/hmc/logs/
#$ -e logs/mcmc/softmax/sim/hmc/logs/

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
conda activate psych_exp_tf-1.15

python3 $BASE_PATH/experiment/research/bnn/proto_bnn_mcmc.py \
    "/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/sim/samples_10/converging/hmc_2e6sssa_5u_0burn_5lag_1e6nr_5e-4ss_3nlfs_1e-2sim_1e1s_res1e4_2020-02-10_18-15-54/data_for_bnn.json" \
    --output_dir "/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/sim/samples_10/converging/hmc_5u_6.7e-3ss_1e5res/" \
    --bnn_weights_file "/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/sim/samples_10/converging/hmc_5u_6.7e-3ss_1e5res/last_weights.json" \
    --num_hidden 5 \
    --cpu 1 \
    --cpu_cores 2 \
    --gpu 0 \
    --kernel_id 'HamiltonianMonteCarlo' \
    --num_results 100000 \
    --step_size 0.0067 \
    --num_leapfrog_steps 3 \
    --lag 10 \
    --scale_identity_multiplier 0.01 \
    --log_level 'INFO' \
    --log_file "$BASE_PATH/logs/mcmc/softmax/sim/hmc/hmc_1.log"

    #--burnin 150000 \
    #--step_adjust_id Simple \
    #--num_adaptation_steps 100000

    # Below only for initial simulation data gen:
    #--random_bnn_init \
    #--num_samples 1000 \
    #--dim 3 \
