#!/bin/bash

#$ -pe smp 24
#$ -N fb_folds_e1
#$ -q debug
#$ -o $HOME/Public/psych_metric/logs/fb/folds/exp1/logs/
#$ -e $HOME/Public/psych_metric/logs/fb/folds/exp1/logs/
#$ -t 1-3

# Exp1 Supplemental Experiment: Data Sensitivity / Bayesian Model Generalization

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
conda activate psych_exp_tf-1.15 

# Constant paths for all runs
BASE_DATA="$HOME/scratch_21/results/sjd/3965_exp_test/fb/fq/folds/all_fold_"
#SRC_PATH="$HOME/scratch_21/results/sjd/3965_exp_test/fb/fq/all_half_1/exp1"

# Unique path addons per run
if [ "$SGE_TASK_ID" -eq "1" ]; then
    SRC_PATH="$BASE_DATA"0/exp1
elif [ "$SGE_TASK_ID" -eq "2" ]; then
    SRC_PATH="$BASE_DATA"1/exp1
elif [ "$SGE_TASK_ID" -eq "3" ]; then
    SRC_PATH="$BASE_DATA"2/exp1
else
    echo "ERROR: Unexpected SGE_TASK_ID: $SGE_TASK_ID"
    exit 1
fi

# Script to execute
python3 "$BASE_PATH/experiment/research/measure/violins.py" \
    "$SRC_PATH/baselines_samples_1.45e4/iid_uniform_dirs/train/euclid_dists_train.csv" \
    "$SRC_PATH/baselines_samples_1.45e4/iid_dirs_adam/train/euclid_dists_train.csv" \
    "$SRC_PATH/baselines_samples_1.45e4/dir-mean_mvn-umvu/train/euclid_dists_train.csv" \
    "$SRC_PATH/baselines_samples_1.45e4/dir-mean_mvn-umvu-zero/train/euclid_dists_train.csv" \
    "$SRC_PATH/4u_1e-1sim/euclid/train/euclid_dists.csv" \
    --test_paths "$SRC_PATH/baselines_samples_1.45e4/iid_uniform_dirs/test/euclid_dists_test.csv" \
    "$SRC_PATH/baselines_samples_1.45e4/iid_dirs_adam/test/euclid_dists_test.csv" \
    "$SRC_PATH/baselines_samples_1.45e4/dir-mean_mvn-umvu/test/euclid_dists_test.csv" \
    "$SRC_PATH/baselines_samples_1.45e4/dir-mean_mvn-umvu-zero/test/euclid_dists_test.csv" \
    "$SRC_PATH/4u_1e-1sim/euclid/test/euclid_dists.csv" \
    --output_path "$SRC_PATH/violins/violins_right_95_cred_lw2.svg" \
    --conditional_models 'Random\nGuess' 'Dirichlet' 'NDoD' 'NDoD Zero' 'BNN' \
    --title 'Facial Beauty: 3 Folds' \
    --inner quartile \
    --measure_lower 0.0 \
    --measure_upper 1.0 \
    --cred_interval_linewidth 2 \
    --cred_intervals_json "$SRC_PATH/cred_95_right_intervals.json"
