"""
Script for running the different truth inference methods.
"""
import argparse
from datetime import datetime
from json import loads as json_loads
import os

import yaml # need to import PyYAML
import numpy as np

# psych metric needs istalled prior to running this script.
# TODO setup the init and everything such that this is accessible once installed
from psych_metric.datasets import data_handler
from psych_metric.truth_inference import truth_inference_model_handler

import random_seed_generator

# TODO create a setup where it is efficient and safe/reliable in saving data
# in a tmp dir as it goes, and saving all of the results in an output directory.

def run_experiments(datasets, models, output_dir, random_seeds):
    """Runs experiments on the datasets using the given random seeds."""

    # Iterates through all datasets and performs the same experiments on them.
    for dataset in datasets:
        # load dataset
        data = data_handler.load_dataset(dataset, encode_columns=True)

        # TODO need to convert into a standard format for all Truth Inference models.
        samples_to_annotators, annotators_to_samples = data.truth_inference_survey_format()

        # Remember to seed the numpy and python random generators, prior to every model running.

        # Iterate through all of the random seeds provided.
        for seed in random_seeds:
            # NOTE I think that the models should be called here... otherwise it is a part of the package to run all of them exhaustively, which does not seem desireable for the package itself.
            #truth_inference.run_models(datasets, models, output_dir, seed)

            # TODO split by [binary/multi]Classification, regression, etc.

            # TODO implement a dataset.task_type == 'regression'|'classification'|'binary_classification'
            if dataset.task_type == 'regression':
                if 'mean' in models:

                if 'median' in models:

            elif 'classification' in dataset.task_type:
                # An Evaluation of Aggregation Technique in Crowdsourcing
                if 'majority_vote' in models:

                if 'majority_decision' in models:

                if 'honeypot' in models:

                if 'ELICE' in models:

                if 'ipierotis_dawid_skene' in models:

                if 'SLME' in models:

                if 'ITER' in models:

                # Comparison of Bayesian Models of Annotation 2018
                if 'multinomial' in models:

                if 'hier_dawid_skene' in models:

                if 'item_diff' in models:

                if 'log_rnd_eff' in models:

                if 'MACE' in models:

                # Truth Inference Survey 2017
                if 'dawid_skene' in models:

                if 'ZenCrowd' in models:

                if 'GLAD' in models:

                if 'minimax' in models:

                if 'BCC' in models:

                if 'CBCC' in models:

                if 'LFC' in models:

                if 'LFC-N' in models:

                if 'CATD' in models:

                if 'PM' in models:

                if 'Multi' in models:

                if 'KOS' in models:

                if 'VI-BP' in models:

                if 'VI-MF' in models:

                # Multi-Class Ground Truth Inference in Crowdsourcing with Clustering 2016
                if 'spectral_dawid_skene' in models:

                if 'GTIC' in models:

    # Could perform summary analysis here, but this is better in post processing

## NOTE the kfold things will only be useful for when we want to experiment with how these perform on subsets of the data. This may be of use when comparing the Truth Inference models relation to ground truth, if there is any connection.
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
    # Check if any arguments are missing in args, if none missing return args.
    if args.output_dir is not None and args.datasets is not None and args.models is not None and args.random_seeds is not None:
        return args

    # Load the configurations from the yaml file at args.config.
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    # Replace all missing values in args with the configuration file values.
    if args.random_seed is None:
        args.random_seed = config['random_seed_file']
    if args.output_dir is None:
        args.output_dir = config['output_dir']

    if args.datasets is None:
        args.datasets = config['datasets']

        # extract any filepaths for the datasets if given.
        for i, dataset in enumerate(args.datasets):
            if isinstance(dataset, dict) and len(dataset) == 1:
                # remove the dict and replace with the dataset identifier
                args.datasets[i] = dataset.keys()[0]

                # dataset filepath given, use that instead of default location.
                if args.datasets_filepaths is None:
                    args.datasets_filepaths = dataset
                elif args.datasets[i] not in args.datasets_filepaths:
                    # only add if it does not exist, otherwise, assume provided as arg.
                    args.datasets_filepaths[args.datasets[i]]= dataset[args.datasets[i]]

    if args.models is None:
        args.models = config['truth_inference']['models']

    return args

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

    # output
    parser.add_argument('-o', '--output_dir', default=None, help='Enter the file path to output directory to store the results.')
    parser.add_argument('--overwrite_output_dir', action='store_true', help='Providing this flag ignores that the output directory already exists and may overwrite existing files.')
    # TODO add datetime to output filenames?
    #parser.add_argument('--datetime_output', action='store_true', help='Providing this flag adds the date and time information to the output as part of the filename.')

    # data
    parser.add_argument('-d', '--datasets', default=None, nargs='+', help='list of the datasets to use ')
    parser.add_argument('-f', '--datasets_filepaths', type=json_loads, help='Dictionary of filepaths to use for certain datasets. Pass in `{"dataset_id": "file/path/"}` to provide the filepath to a dataset.')

    parser.add_argument('-m', '--models', default=None, nargs='+', help='Truth inference models to test.')

    # Random seeds and itereations of models
    parser.add_argument('r',  '--random_seeds', default=None, help='The filepath to the file that contains the random seeds to use for the tests.')
    parser.add_argument('i', '--iterations', default=None, type=int, help='The number of iterations to run the set of models for on the data and also the number of random seeds that will be generated in output file. This is ignored if random_seeds is provided via arguments or the configuration file.')

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
    for dataset in args.datasets:
        if not data_handler.dataset_exists(dataset):
            raise UserWarning('Unrecognized dataset `%s`. This dataset will be ignored'%dataset)
            unrecognized_datasets.add(dataset)

            # Remove the dataset from the datasets_filepaths dict, if present
            args.datasets_filepaths.pop(dataset, None)

    # Remove unrecognized datasets
    args.datasets = set(args.datasets) - unrecognized_datasets
    if len(args.datasets) <= 0:
        raise Exception('There are no remaining datasets after removing the unrecognized datasets.')

    # TODO Confirm that the truth inference models are valid.
    if args.models is None:
        raise Exception('No models provided.')

    unrecognized_models = set()
    for model in args.models:
        if not truth_inference_model_handler.model_exists(model):
            raise UserWarning('Unrecognized model `%s`. This model will be ignored'%model)
            unrecognized_models.add(model)

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
