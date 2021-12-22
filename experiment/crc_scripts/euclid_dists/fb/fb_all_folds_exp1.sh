#!/bin/bash

#$ -pe smp 4
#$ -N fb_folds_e1
#$ -q debug
#$ -o $HOME/Public/psych_metric/logs/fb/folds/exp1/logs/
#$ -e $HOME/Public/psych_metric/logs/fb/folds/exp1/logs/
#$ -t 1-6

# Exp1 Supplemental Experiment: Data Sensitivity / Bayesian Model Generalization

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
conda activate psych_exp_tf-1.15 

# Constant paths for all runs
BASE_BNN_WEIGHTS="$HOME/scratch_21/results/debug/bnn_softmax/fb/4u_1e-1/folds/all"
BASE_DATA_JSON="$HOME/scratch_21/under_over/fb_fq_16b_under_over/fb_full_trained/folds/proto_bnn_format_all_fold_"
BASE_OUTPUT_DIR="$HOME/scratch_21/results/sjd/3965_exp_test/fb/fq/folds/all_fold_"

# Unique path addons per run
if [ "$SGE_TASK_ID" -eq "1" ]; then
    BNN_WEIGHTS="$BASE_BNN_WEIGHTS/fold_0/sampling/hmc_4u_1e-1sim_3nlfs_0lag_1e6res/0.0029ss/2e7burn/s2e4_accepted.json"
    DATA_JSON="$BASE_DATA_JSON"0_train.json
    OUTPUT_DIR="$BASE_OUTPUT_DIR"0/exp1/4u_1e-1sim/euclid/train
elif [ "$SGE_TASK_ID" -eq "2" ]; then
    BNN_WEIGHTS="$BASE_BNN_WEIGHTS/fold_1/sampling/hmc_4u_1e-1sim_3nlfs_0lag_1e6res/0.0014ss/2e7burn/s2e4_accepted.json"
    DATA_JSON="$BASE_DATA_JSON"1_train.json
    OUTPUT_DIR="$BASE_OUTPUT_DIR"1/exp1/4u_1e-1sim/euclid/train
elif [ "$SGE_TASK_ID" -eq "3" ]; then
    BNN_WEIGHTS="$BASE_BNN_WEIGHTS/fold_2/sampling/hmc_4u_1e-1sim_3nlfs_0lag_1e6res/0.0027ss/2e7burn/s2e4_accepted.json"
    DATA_JSON="$BASE_DATA_JSON"2_train.json
    OUTPUT_DIR="$BASE_OUTPUT_DIR"2/exp1/4u_1e-1sim/euclid/train
elif [ "$SGE_TASK_ID" -eq "4" ]; then
    BNN_WEIGHTS="$BASE_BNN_WEIGHTS/fold_0/sampling/hmc_4u_1e-1sim_3nlfs_0lag_1e6res/0.0029ss/2e7burn/s2e4_accepted.json"
    DATA_JSON="$BASE_DATA_JSON"0_test.json
    OUTPUT_DIR="$BASE_OUTPUT_DIR"0/exp1/4u_1e-1sim/euclid/test
elif [ "$SGE_TASK_ID" -eq "5" ]; then
    BNN_WEIGHTS="$BASE_BNN_WEIGHTS/fold_1/sampling/hmc_4u_1e-1sim_3nlfs_0lag_1e6res/0.0014ss/2e7burn/s2e4_accepted.json"
    DATA_JSON="$BASE_DATA_JSON"1_test.json
    OUTPUT_DIR="$BASE_OUTPUT_DIR"1/exp1/4u_1e-1sim/euclid/test
elif [ "$SGE_TASK_ID" -eq "6" ]; then
    BNN_WEIGHTS="$BASE_BNN_WEIGHTS/fold_2/sampling/hmc_4u_1e-1sim_3nlfs_0lag_1e6res/0.0027ss/2e7burn/s2e4_accepted.json"
    DATA_JSON="$BASE_DATA_JSON"2_test.json
    OUTPUT_DIR="$BASE_OUTPUT_DIR"2/exp1/4u_1e-1sim/euclid/test
else
    echo "ERROR: Unexpected SGE_TASK_ID: $SGE_TASK_ID"
    exit 1
fi

# Script to execute
python3 "$BASE_PATH/experiment/research/measure/euclid_dist_bnn.py" \
    "$DATA_JSON" \
    --bnn_weights_file "$BNN_WEIGHTS" \
    --output_dir "$OUTPUT_DIR" \
    --num_layers 1 \
    --num_hidden 4 \
    --normalize \
    --cpu_cores 1 \
    --gpu 0 \
    --log_file "$BASE_PATH/logs/fb/folds/exp1/l2.log" \
    --log_level 'INFO'

python3 "$BASE_PATH/experiment/research/measure/col_means.py" \
    "$OUTPUT_DIR/euclid_dists.csv" \
    "$OUTPUT_DIR/euclid_dists_means_per_weight_sets.csv" \

python3 "$BASE_PATH/experiment/research/measure/col_means.py" \
    "$OUTPUT_DIR/euclid_dists.csv" \
    "$OUTPUT_DIR/euclid_dists_means_per_sample.csv" \
    --axis 1
