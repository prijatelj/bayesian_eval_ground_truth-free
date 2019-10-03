"""Run the distribution tests."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime
import json
import logging
import os

import numpy as np
import tensorflow as tf

from psych_metric import distribution_tests


def parse_args():
        parser = argparse.ArgumentParser(description='Run proof of concept ')

    # Model args.
    parser.add_argument(
        '-m',
        '--model_id',
        default='vgg16',
        help='The model to use',
        choices=['vgg16', 'resnext50'],
    )
    parser.add_argument(
        '-p',
        '--parts',
        default='labelme',
        help='The part of the model to use, if parts are allowed (vgg16)',
        choices=['full', 'vgg16', 'labelme'],
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        default=32,
        type=int,
        help='The number of units in dense layer of letnet.'
    )
    parser.add_argument(
        '-e',
        '--epochs',
        default=1,
        type=int,
        help='The number of epochs.',
    )
    parser.add_argument(
        '-r',
        '--random_seeds',
        default=None,
        nargs='+',
        type=int,
        help='The random seed to use for initialization of the model.',
    )
    parser.add_argument(
        '-w',
        '--weights_file',
        default=None,
        help='The file containing the model weights to set at initialization.',
    )
    parser.add_argument(
        '--kl_div',
        action='store_true',
        help='Uses Kullback Leibler Divergence as loss instead of Categorical Cross Entropy',
    )

    # Data args
    parser.add_argument(
        '-d',
        '--dataset_id',
        default='LabelMe',
        help='The dataset to use',
        choices=['LabelMe', 'FacialBeauty', 'All_Ratings'],
    )
    parser.add_argument(
        'dataset_filepath',
        help='The filepath to the data directory',
    )
    parser.add_argument(
        '-l',
        '--label_src',
        default='majority_vote',
        help='The source of labels to use for training.',
        choices=['majority_vote', 'frequency', 'ground_truth', 'annotations'],
    )
    parser.add_argument(
        '--focus_fold',
        default=None,
        type=int,
        help='The focus fold to split the data on to form train and test sets for a singlemodel train and evaluate session (No K-fold Cross Validation; Just evaluates one partition).',
    )

    # Output args
    parser.add_argument(
        '-o',
        '--output_dir',
        default='./',
        help='Filepath to the output directory.',
    )
    parser.add_argument(
        '--tb_summary_dir',
        default=None,
        help='Filepath appened to the output directory for saving the summaries.',
    )
    parser.add_argument(
        '-s',
        '--summary_path',
        default='',
        help='Filepath appened to the output directory for saving the summaries.',
    )

    # Hardware
    parser.add_argument(
        '--cpu',
        default=1,
        type=int,
        help='The number of available CPUs.',
    )
    parser.add_argument(
        '--cpu_cores',
        default=1,
        type=int,
        help='The number of available cores per CPUs.',
    )
    parser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help='The number of available GPUs. Pass negative value if no CUDA.',
    )
    parser.add_argument(
        '--which_gpu',
        default=None,
        type=int,
        help='The number of available GPUs. Pass negative value if no CUDA.',
    )

    # K Fold CV args
    parser.add_argument(
        '-k',
        '--kfolds',
        default=5,
        type=int,
        help='The number of available CPUs.',
    )
    parser.add_argument(
        '--no_shuffle_data',
        action='store_false',
        help='Disable shuffling of data.',
    )
    parser.add_argument(
        '--crowd_layer',
        action='store_true',
        help='Use crowd layer in ANNs.',
    )
    parser.add_argument(
        '--no_save_pred',
        action='store_false',
        help='Predictions will not be saved.',
    )
    parser.add_argument(
        '--no_save_model',
        action='store_false',
        help='Model will not be saved.',
    )
    parser.add_argument(
        '--stratified',
        action='store_true',
        help='Stratified K fold cross validaiton will be used.',
    )
    parser.add_argument(
        '--train_focus_fold',
        action='store_true',
        help='The focus fold in K fold cross validaiton will be used for '
        + 'training and the rest will be used for testing..',
    )

    parser.add_argument(
        '--period',
        default=0,
        type=int,
        help='The number of epochs between checkpoints for ModelCheckpoint.',
    )

    parser.add_argument(
        '--period_save_pred',
        action='store_true',
        help='Saves trained models performance on validation data for every period.',
    )

    # Logging
    parser.add_argument(
        '--log_level',
        default='WARNING',
        help='The log level to be logged.',
    )
    parser.add_argument(
        '--log_file',
        default=None,
        type=str,
        help='The log file to be written to.',
    )

    args = parser.parse_args()

    # Post-handling of args as necessary

    return args


if __name__ == '__main__':
    args = parse_args()
