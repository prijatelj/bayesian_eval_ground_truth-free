"""
Script for running the different truth inference methods.
"""
import argparse
import os
import yaml
import csv
from datetime import datetime

import numpy as np

# psych metric needs istalled prior to running this script.
# TODO setup the init and everything such that this is accessible once installed
from psych_metric.datasets import data_handler

# TODO create a setup where it is efficient and safe/reliable in saving data
# in a tmp dir as it goes, and saving all of the results in an output directory.

import random_seed_generator

def run_experiments(datasets, random_seeds_file, K=1, R=1):
    """Performs the set of experiments on the given datasets."""

    # Iterates through all datasets and performs the same experiments on them.
    for dataset in datasets:
        # seed the numpy random generator
        np.random.seed(seed)

        # load dataset
        data = data_handler.load_dataset(dataset, encode_columns=True)

        # TODO need to convert into a standard format for all Truth Inference models.
        # Either an annotator (sparse) matrix rows as samples or annotator list

        # Perform evaluation experiments on dataset.
        r_looped_kfold_eval(X, y, k, n)

        # Perform train, test split experiments on dataset, if train, test provided.

def r_looped_kfold_eval(X, y, K=10, N=1, truth_inference_models=None, seed=None, results_dir=None):
    """Performs N looped Kfold cross validaiton on the provided data and returns
    the resulting information unless told to save the data as it runs.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The input data
    y : array-like, shape (n_samples) or (n_samples, n_outputs)
        The target values.
    N : int, optional
        number of iterations to run the K fold cross validation to provide more
        accuracte results by overcoming the variace due to the random split used
        for the K fold cross validation.
    K : int, optional
        number of folds to use in K fold cross validation
    seed : {None, int}, optional
        Random seed used to initialize the pseudo-random number generator.
    results_dir : str, optional
        Saves the results in the given directory as it progresses.

    Returns
    -------
    dict
        A dictionary of the experiment results, or None if told to be memory
        fficient and instead save the results to the filesystem.
    """
    # TODO loop through the random_seeds rather than R.
    for r in range(R):
        kfold_Eval(X, y, K, truth_inference_models, seed, results_dir)

def kfold_eval(X, y, K=10, truth_inference_models=None, seed=None, results_dir=None):
        skf = StratifiedKFold(K, True, random_state)

        for k, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Compute the given models on the provided data.

            if results_dir is not None:
                # Save results of this fold

        if results_dir is not None:
            # Save results of this iteration of the N loop

def load_config(args):
    """Loads the settings from the given configuration file into the argparse
    namespace for the missing arguments.

    Returns
    -------
        argparse namespace with the orignally missing values filled from the
        configuration file.
    """
    # TODO load the configurations from the yaml file at args.config.

    return

def parse_args():
    """Parses the arguments when this script is called and sets up the
    experiment variables based on the provided configuration file and arguments.

    Returns
    -------
    argparse.Namespace
        The configuration variables for the experiments to be run.
    """
    parser = argparse.ArgumentParser(description='Run a set of truth inference models on the provided datasets and save their output.')

    parser.add_argument('-c', '--config', default=None, help='Enter the file path to the configuration yaml file.')

    parser.add_argument('-o', '--output_dir', default=None, help='Enter the file path to output directory to store the results.')
    parser.add_argument('--overwrite_output_dir', action='store_true', help='Providing this flag ignores that the output directory already exists and may overwrite existing files.')
    parser.add_argument('-d','--datasets', default=None, help='')
    parser.add_argument('-m','--models', default=None, help='Truth inference models to test.')
    parser.add_argument('r', '--random_seeds', default=None, help='The filepath to the file that contains the random seeds to use for the tests.')
    parser.add_argument('i','--iterations', default=None, type=int, help='The number of iterations to run the set of models for on the data and also the number of random seeds that will be generated in output file. This is ignored if random_seeds is provided via arguments or the configuration file.')

    args =  parser.parse_args()

    # TODO Ensure all arguments are valid and load those missing from config
    if args.config is None and (args.datasets is None or args.models is None or output_dir is None):
        raise Exception('Must provide a configuration file via the config flag or pass all arguments. when calling this program.')
    elif args.config is not None:
        # Load the values from the config file into args that are not present in args
        args = load_config(args)

    # Confirm output destination is valid and create if does not exist.
    if args.output_dir is None:
        raise Exception('No output directory provided.')
    elif os.path.is_file(args.output_dir):
        raise Exception('`%s` is a file, not a directory.'%args.output_dir)
    else:
        # attempt to create the output directory
        if args.overwrite_output_dir:
            raise UserWarning('The `overwrite_output_dir` argument has been passed. The provided output directory exists and this program may overwrite existing files if the generated results share the same filename.')

        # if already exists then raise an error, unless overwrite_output_dir
        os.makedirs(args.output_dir, exist_ok=args.overwrite_output_dir)

    # Ensure the random seeds were provided, or create them
    if isinstance(args.random_seeds, str) and os.path.is_file(args.random_seeds):
        # Random seeds file exists, load the random seeds from file.
        with open(args.random_seeds, 'r') as random_seeds_file:
            args.random_seeds = []
            for seed in random_seeds_file:
                args.random_seeds.append(int(seed))

    elif args.random_seeds is None and isinstance(args.iterations, int):
        raise UserWarning('A random seeds file was not provided. The random seeds wll be generate for every iteration, totaling %(iter)d random seeds. A file containing these seeds will be saved along with the output at `%(output)s/random_seeds_count-%(iter)d.txt`.' % {'iter':args.iterations, 'output':args.output_dir})
        args.random_seeds = random_seed_generator.generate(args.iterations)

        # Save the newly generated random seeds to a file:
        random_seed_generator.save_to_file(args.random_seeds, os.path.join(args.output_dir, 'random_seeds_count-%d.txt'%args.iterations))

    elif args.random_seeds is None and not isinstance(args.iterations, int):
        raise Exception('The random seeds file was not provided and nor was the number of desired iterations. The random seeds were unable to be generated. Please provide either a random seeds file or a number of iterations.')

    # Confirm datasets are valid
    if args.datasets is None:
        raise Exception('No datasets provided.')

    unrecognized_datasets = set()
    for d in args.datasets:
        if not data_handler.dataset_exists(dataset):
            raise UserWarning('Unrecognized dataset `%s`. This dataset will be ignored'%d)
            unrecognized_datasets.add(d)

    # Remove unrecognized datasets
    args.datasets = set(args.datasets) - unrecognized_datasets
    if len(args.datasets) <= 0:
        raise Exception('There are no remaining datasets after removing the unrecognized datasets.')

    # TODO Confirm that the truth inference models are valid.
    if args.models is None:
        raise Exception('No models provided.')

    unrecognized_models = set()
    for d in args.models:
        if not data_handler.dataset_exists(dataset):
            raise UserWarning('Unrecognized dataset `%s`. This dataset will be ignored'%d)
            unrecognized_models.add(d)

    # Remove unrecognized truth inference models
    args.models = set(args.models) - unrecognized_models
    if len(args.models) <= 0:
        raise Exception('There are no remaining models after removing the unrecognized models.')

    return args

if __name__ == '__main__':
    # Read in arguements and configuaration file
    args = parse_args()

    # TODO Run the experiments,
    run_experiments()
