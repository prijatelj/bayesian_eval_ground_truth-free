"""
Script for running the experiements of the different truth inference models and
metric comparisons.
"""
import argparse
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold

# psych metric needs istalled prior to running this script.
#import psych_metric

# TODO create a setup where it is efficient and safe/reliable in saving data
# in a tmp dir as it goes, and saving all of the results in an output directory.

# TODO probably need a set of seeds such that the random values throughout are
# reproducible and independent of the sequential runs of the random_state.

def experiments(datasets, random_seeds_file, K=1, N=1):
    """Performs the set of experiments on the given datasets."""

    # Iterates through all datasets and performs the same experiments on them.
    for dataset in datasets:
        # seed the numpy random generator
        np.random.seed(seed)

        # load dataset
        data = psych_metric.datasets.load_dataset(dataset, encode_columns=True)

        # TODO need to convert into a standard format for all Truth Inference models.
        # Either an annotator (sparse) matrix rows as samples or annotator list

        # Perform evaluation experiments on dataset.
        n_nested_kfold_eval(X, y, k, n)

        # Perform train, test split experiments on dataset, if train, test provided.

def n_nested_kfold_eval(X, y, K=10, N=1, truth_inference_models=None, seed=None, results_dir=None):
    """Performs N nested Kfold cross validaiton on the provided data and returns
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
    for n in range(N):
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
def parse_args():
    """Parses the arguments when this script is called and sets up the experiment
    variables based on the provided configuration file and arguments.

    Returns
    -------
    argparse.Namespace
        The configuration variables for the experiments to be run.
    """
    parser = argparse.ArgumentParser(description='Convert the LEDA graph representation into an adj matrix.')

    parser.add_argument('-c', '--config', default='config.yaml', help='Enter the file path to the configuration file.')

    parser.add_argument('-o', '--output_dir', default=None, help='Enter the file path to output directory where all the results are stored.')

    parser.add_argument('-o', '--output_dir', default=None, help='Enter the file path to output file.')
    parser.add_argument('-d','--datasets', default=None, help='')
    parser.add_argument('--models', default=None, help='Truth inference models to test.')
    parser.add_argument('--metrics', default=None, help='Comparision metrics to use and test.')
    parser.add_argument('--seed', default=None, help='Global initial seed that allows for more reproducible experiments.')

    args =  parser.parse_args()

    # TODO Ensure all arguments are valid
    # TODO confirm datasets are present
    # TODO confirm output destination is valid

    return args

if __name__ == '__main__':
    # TODO Read in arguements and configuaration file
    args = parse_args()

    # Setup anythiing necessary prior to running the experiments
    # such as, create main directories

    # TODO Run the experiments,
