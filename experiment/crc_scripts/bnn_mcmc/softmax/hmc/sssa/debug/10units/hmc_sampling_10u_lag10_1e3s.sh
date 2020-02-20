#!/bin/bash

#$ -pe smp 4
#$ -N Sim10u1e3sS
#$ -q debug
#$ -o logs/mcmc/softmax/sim/hmc_bnn_10units/sampling/logs/
#$ -e logs/mcmc/softmax/sim/hmc_bnn_10units/sampling/logs/
#$ -t 1-50

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
conda activate psych_exp_tf-1.15

# Uses same 1000 points as 5 hiddent unit version
python3 $BASE_PATH/experiment/research/bnn/proto_bnn_mcmc.py \
    "/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/sim/samples_1000/sssa/initial_run_gen_data/data_for_bnn.json" \
    --output_dir "/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/sim/samples_1000/10units/sampling/hmc_10u_9.15e3lag_6.8e-4ss_1e3res/job_$SGE_TASK_ID/" \
    --bnn_weights_file "/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/sim/samples_1000/10units/converging/hmc_10u_1e1lag_6.8e-4ss_1.5e4burn_1e5res/_2020-02-18_22-47-06/last_weights.json" \
    --num_hidden 10 \
    --cpu 1 \
    --cpu_cores 4 \
    --gpu 0 \
    --kernel_id 'HamiltonianMonteCarlo' \
    --num_results 1000 \
    --step_size 0.00068 \
    --num_leapfrog_steps 3 \
    --lag 9150 \
    --burnin 9150 \
    --parallel_chains 4 \
    --scale_identity_multiplier 0.01 \
    --log_level 'INFO' \
    --log_file "$BASE_PATH/logs/mcmc/softmax/sim/hmc_bnn_10units/sampling/hmc_$SGE_TASK_ID.log"
