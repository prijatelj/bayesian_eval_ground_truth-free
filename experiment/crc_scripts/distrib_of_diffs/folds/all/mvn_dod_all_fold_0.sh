#!/bin/bash

#$ -pe smp 4
#$ -N fbDoD_1.45e4
#$ -q long
#$ -o $HOME/Public/psych_metric/logs/mcmc/softmax/fb/fq/dod/logs/
#$ -e $HOME/Public/psych_metric/logs/mcmc/softmax/fb/fq/dod/logs/

# set up the environment
module add conda
conda activate psych_exp_tf-1.15

python3 /afs/crc.nd.edu/user/d/dprijate/Public/psych_metric/experiment/research/measure/sjd_euclid_dist.py \
    "$HOME/scratch_21/under_over/fb_fq_16b_under_over/fb_full_trained/folds/proto_bnn_format_all_fold_0_train.json" \
    --test_datapath '$HOME/scratch_21/under_over/fb_fq_16b_under_over/fb_full_trained/folds/proto_bnn_format_all_fold_0_test.json' \
    --output_dir "$HOME/scratch_21/results/sjd/3965_exp_test/fb/fq/folds/all_fold_0/exp1/baselines_samples_1.45e4/" \
    --sample_size 14500 \
    --cpu_cores 4 \
    --gpu 0 \
    --src_candidates 'dir-mean_mvn-umvu'\
    --save_conds \
    --log_level INFO \
    --normalize
