#!/bin/bash

#$ -pe smp 4
#$ -N fbH_KLD_all
#$ -q debug
#$ -o $HOME/Public/psych_metric/logs/fb/all_half_1/exp2/kldiv/logs/
#$ -e $HOME/Public/psych_metric/logs/fb/all_half_1/exp2/kldiv/logs/
#$ -t 1-5

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
conda activate psych_exp_tf-1.15 

# Constant paths for all runs
BNN_WEIGHTS="$HOME/scratch_21/results/debug/bnn_softmax/fb/4u_1e-1/all_half_1/sampling/hmc_4u_1e-1sim_3nlfs_3310lag_burn_2.8e-3ss_1e6res/s2e4_accepted.json"

BASE_DATA_JSON="$HOME/scratch_21/under_over/fb_fq_16b_under_over/fb_full_trained/"

BASE_OUTPUT_DIR="$HOME/scratch_21/results/sjd/3965_exp_test/fb/fq/all_half_1/exp2/4u_1e-1sim/kldiv/"

# Unique path addons per run
if [ "$SGE_TASK_ID" -eq "1" ]; then
    # train
    DATA_JSON="$BASE_DATA_JSON/proto_bnn_format_train.json"
    OUTPUT_DIR="$BASE_OUTPUT_DIR/train/"
elif [ "$SGE_TASK_ID" -eq "2" ]; then
    # test
    DATA_JSON="$BASE_DATA_JSON/proto_bnn_format_test.json"
    OUTPUT_DIR="$BASE_OUTPUT_DIR/test/"
elif [ "$SGE_TASK_ID" -eq "3" ]; then
    # all
    DATA_JSON="$BASE_DATA_JSON/proto_bnn_format_all.json"
    OUTPUT_DIR="$BASE_OUTPUT_DIR/all/"
elif [ "$SGE_TASK_ID" -eq "4" ]; then
    # all half 1
    DATA_JSON="$BASE_DATA_JSON/proto_bnn_format_all_half_1.json"
    OUTPUT_DIR="$BASE_OUTPUT_DIR/all_half_1/"
elif [ "$SGE_TASK_ID" -eq "5" ]; then
    # all half 2
    DATA_JSON="$BASE_DATA_JSON/proto_bnn_format_all_half_2.json"
    OUTPUT_DIR="$BASE_OUTPUT_DIR/all_half_2/"
else
    echo "ERROR: Unexpected SGE_TASK_ID: $SGE_TASK_ID"
    exit 1
fi

# Script to execute
python3 "$BASE_PATH/experiment/research/measure/kldiv.py" \
    "$DATA_JSON" \
    --bnn_weights_file "$BNN_WEIGHTS" \
    --output_dir "$OUTPUT_DIR" \
    --num_layers 1 \
    --num_hidden 4 \
    --cpu_cores 4 \
    --gpu 0 \
    --log_file "$BASE_PATH/logs/fb/all_half_1/exp2/kldiv.log" \
    --log_level 'INFO'

python3 "$BASE_PATH/experiment/research/measure/col_means.py" \
    "$OUTPUT_DIR/kldivergence.csv" \
    "$OUTPUT_DIR/kldivergence_means_per_weight_sets.csv"
