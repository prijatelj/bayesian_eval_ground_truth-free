#!/bin/bash

#$ -pe smp 4
#$ -N fb_raw_e3
#$ -q long
#$ -o $HOME/Public/psych_metric/logs/fb/train_test/raw_exp3/logs/
#$ -e $HOME/Public/psych_metric/logs/fb/train_test/raw_exp3/logs/
#$ -t 1-8

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module add conda
conda activate psych_exp_tf-1.15 

# Constant paths for all runs
BASE_DATA="$HOME/scratch_21/under_over/fb_fq_16b_under_over"

# Unique path addons per run
if [ "$SGE_TASK_ID" -eq "1" ]; then
    # untrained
    DATA_JSON="$BASE_DATA/fb_untrained/untrained_fb_pred_train.json"
    OUTPUT_DIR="$BASE_DATA/fb_untrained/raw/train"
elif [ "$SGE_TASK_ID" -eq "2" ]; then
    # underfit
    DATA_JSON="$BASE_DATA/fb_half/fb_3ep_train.json"
    OUTPUT_DIR="$BASE_DATA/fb_half/raw/train"
elif [ "$SGE_TASK_ID" -eq "3" ]; then
    # full fit
    DATA_JSON="$BASE_DATA/fb_full_trained/proto_bnn_format_train.json"
    OUTPUT_DIR="$BASE_DATA/fb_full_trained/raw/train"
elif [ "$SGE_TASK_ID" -eq "4" ]; then
    # overfit
    DATA_JSON="$BASE_DATA/fb_over/fb_60ep_train.json"
    OUTPUT_DIR="$BASE_DATA/fb_over/raw/train"
elif [ "$SGE_TASK_ID" -eq "5" ]; then
    # untrained
    DATA_JSON="$BASE_DATA/fb_untrained/untrained_fb_pred_test.json"
    OUTPUT_DIR="$BASE_DATA/fb_untrained/raw/test"
elif [ "$SGE_TASK_ID" -eq "6" ]; then
    # underfit
    DATA_JSON="$BASE_DATA/fb_half/fb_3ep_test.json"
    OUTPUT_DIR="$BASE_DATA/fb_half/raw/test"
elif [ "$SGE_TASK_ID" -eq "7" ]; then
    # full fit
    DATA_JSON="$BASE_DATA/fb_full_trained/proto_bnn_format_test.json"
    OUTPUT_DIR="$BASE_DATA/fb_full_trained/raw/test"
elif [ "$SGE_TASK_ID" -eq "8" ]; then
    # overfit
    DATA_JSON="$BASE_DATA/fb_over/fb_60ep_test.json"
    OUTPUT_DIR="$BASE_DATA/fb_over/raw/test"
else
    echo "ERROR: Unexpected SGE_TASK_ID: $SGE_TASK_ID"
    exit 1
fi

# Script to execute
python3 "$BASE_PATH/experiment/research/measure/raw_measure.py" \
    "$DATA_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --normalize \
    --measure all \
    --log_file "$BASE_PATH/logs/fb/raws_exp3/l2.log" \
    --log_level 'INFO'
