#!/bin/bash

#$ -pe smp 4        # Specify parallel environment and legal core size
#$ -N lab_mv       # Specify job name
#$ -q gpu
#$ -l gpu_card=1
#$ -o logs/label_me/mv/logs/
#$ -e logs/label_me/mv/logs/
#$ -t 1-50

BASE_PATH="$HOME/Public/psych_metric"

# set up the environment
module load cudnn cuda tensorflow
conda activate metric

# get the random seed for the specific job from file.
SEED="$(sed "$SGE_TASK_ID q;d" $BASE_PATH/experiment/random_seeds/random_seeds_count-100.txt)"

python3 "$BASE_PATH/predictors.py" \
    "$BASE_PATH/psych_metric/datasets/crowd_layer/crowd_layer_data/" \
    --cpu 1 \
    --cpu_cores 4 \
    --gpu 1 \
    --log_level INFO \
    --log_file "logs/label_me/mv/$SEED-vgg16.log" \
    --output_dir "$BASE_PATH/results/predictors/early_stop/mv/" \
    --random_seeds "$SEED" \
    --epochs 50 \
    --batch_size 64 \
    --model_id vgg16 \
    --dataset_id LabelMe \
    --label_src majority_vote \
    --parts labelme \
    --patience 5 \
    --restore_best_weights
