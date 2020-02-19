#!/bin/bash

#$ -pe smp 2
#$ -N Sim10u1e3s
#$ -q debug
#$ -o logs/mcmc/softmax/sim/hmc_bnn_10units/logs/
#$ -e logs/mcmc/softmax/sim/hmc_bnn_10units/logs/

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
conda activate psych_exp_tf-1.15

# Uses same 1000 points as 5 hiddent unit version
python3 $BASE_PATH/experiment/research/bnn/proto_bnn_mcmc.py \
    "/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/sim/samples_1000/sssa/initial_run_gen_data/data_for_bnn.json" \
    --output_dir "/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/sim/samples_1000/10units/converging/hmc_10u_1e1lag_6.8e-4ss_1.5e4burn_1e5res/" \
    --bnn_weights_file "/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/sim/samples_1000/10units/converging/hmc_10u_1e1lag_6.8e-4ss_1.5e4burn_1e5res/_2020-02-18_22-47-06/last_weights.json" \
    --num_hidden 10 \
    --cpu 1 \
    --cpu_cores 2 \
    --gpu 0 \
    --kernel_id 'HamiltonianMonteCarlo' \
    --num_results 100000 \
    --step_size 0.00068 \
    --num_leapfrog_steps 3 \
    --lag 10 \
    --scale_identity_multiplier 0.01 \
    --log_level 'INFO' \
    --log_file "$BASE_PATH/logs/mcmc/softmax/sim/hmc_bnn_10units/hmc_1.log" \
    --burnin 15000
    #--step_adjust_id Simple \
    #--num_adaptation_steps 150000
