#!/bin/bash

#$ -pe smp 4
#$ -N Sim5u1e3sS
#$ -q debug
#$ -o logs/mcmc/softmax/sim/hmc/5u1e3/sampling/logs/
#$ -e logs/mcmc/softmax/sim/hmc/5u1e3/sampling/logs/
#$ -t 1-50

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
conda activate psych_exp_tf-1.15

python3 $BASE_PATH/experiment/research/bnn/proto_bnn_mcmc.py \
    "/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/sim/samples_1000/sssa/initial_run_gen_data/data_for_bnn.json" \
    --output_dir "/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/sim/samples_1000/sampling/hmc_5u_8e-4ss_39670lag_1e3res/s2e4/job_$SGE_TASK_ID/" 
    --bnn_weights_file "/afs/crc.nd.edu/group/cvrl/scratch_21/dprijate/results/debug/bnn_softmax/sim/samples_1000/converging/hmc_5u_8e-4ss_1e5res/last_weights.json" \
    --num_hidden 5 \
    --cpu 1 \
    --cpu_cores 4 \
    --gpu 0 \
    --kernel_id 'HamiltonianMonteCarlo' \
    --num_results 100 \
    --step_size 0.0008 \
    --num_leapfrog_steps 3 \
    --lag 39670 \
    --burnin 39670 \
    --parallel_chains 4 \
    --scale_identity_multiplier 0.01 \
    --log_level 'INFO' \
    --log_file "$BASE_PATH/logs/mcmc/softmax/sim/hmc/5u1e3/sampling/hmc_$SGE_TASK_ID.log"
