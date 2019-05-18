"""
Script for running the different truth inference methods.
"""
import argparse
from datetime import datetime
from json import loads as json_loads
import os
from time import perf_counter, process_time
import warnings

import yaml # need to import PyYAML
#import numpy as np
import pandas as pd

# psych metric needs istalled prior to running this script.
# TODO setup the init and everything such that this is accessible once installed
from psych_metric.datasets import data_handler
from psych_metric.truth_inference import truth_inference_model_handler, zheng_2017

import random_seed_generator

# TODO create a setup where it is efficient and safe/reliable in saving data
# in a tmp dir as it goes, and saving all of the results in an output directory.

def summary_csv(filename, truth_inference_method, parameters, dataset_id, dataset_filepath, random_seed, runtime_process, runtime_performance, datetime_start, datetime_end):
    """Convenience function creates a summary csv file at the given path
    containing the provided arguments.

    Parameters
    ----------
    """
    with open(filename, 'w') as f:
        f.write('truth_inference_method,%s\n' % truth_inference_method)
        f.write('parameters,%s\n' % repr(parameters))
        f.write('dataset,%s\n' % dataset_id)
        f.write('dataset_filepath,%s\n' % dataset_filepath)
        f.write('random_seed,%d\n' % random_seed)
        f.write('runtime_process,%f\n' % runtime_process)
        f.write('runtime_performance,%f\n' % runtime_performance)
        f.write('datetime_start,%s\n' % str(datetime_start))
        f.write('datetime_end,%s' % str(datetime_end))


def zheng_2017_models_exist(models):
    """Checks if any of the zheng 2017 models are present in provided models
    set."""
    return


def run_experiments(datasets, models, output_dir, random_seeds, datasets_filepaths=None, print_progress=True):
    """Runs experiments on the datasets using the given random seeds.

    Parameters
    ----------
    datasets : list(str)
        List of string identifiers for the datasets to be loaded and used for
        the truth inference task.
    models : list(str) | dict(str:dict())
        List of string identifiers for the truth inference models to be used.
        Or, this is a dictionary of string identifiers for the truth inference
        models to be used and they map to a dictionary of parameters for each
        model. OR a list of dictionaries of parameters to be tested for every
        random seed.
    output_dir : str
        String of the filepath to the root output directory for saving results.
    random_seeds : list(int)
        List of integer random seeds to use for initializing each test of the
        truth inference methods.
    dataset_filepaths : dict(str:str)
        Dictionary with keys as string dataset identifiers and values as string
        dataset filepaths. If a dataset is not in this dictionary, then it's
        filepath is not provided.
    """

    # Iterates through all datasets and performs the same experiments on them.
    for i, dataset_id in enumerate(datasets):
        # Progress print out
        if print_progress:
            print('%d / %d datasets. Current dataset: %s' % (i+1, len(datasets), dataset_id))

        # Get dataset filepath if given
        dataset_filepath = datasets_filepaths[dataset_id] if datasets_filepaths is not None and dataset_id in datasets_filepaths else None
        # load dataset
        #data = data_handler.load_dataset(dataset_id, dataset_filepath, encode_columns=True)
        dataset = data_handler.load_dataset(dataset_id, dataset_filepath)

        # TODO Make a check if the Zheng 2017 models are in models.
        #if zheng_2017_models_exist(models):
        samples_to_annotators, annotators_to_samples = dataset.truth_inference_survey_format()

        # Save the parent/data collection name of dataset for result output dir
        if data_handler.dataset_exists(dataset_id, 'truth_survey_2017'):
            collection = 'truth_survey_2017'
        elif data_handler.dataset_exists(dataset_id, 'snow_2008'):
            collection = 'snow_2008'
        elif data_handler.dataset_exists(dataset_id, 'ipeirotis_2010'):
            collection = 'ipeirotis_2010'
        elif data_handler.dataset_exists(dataset_id, 'facial_beauty'):
            collection = 'facial_beauty'
        elif data_handler.dataset_exists(dataset_id, 'crowd_layer'):
            collection = 'crowd_layer'
        else:
            collection = None

        output_data_dir = os.path.join(output_dir, dataset_id) if collection is None else os.path.join(output_dir, collection, dataset_id)
        # Make the  directory structure if it does not already exist.
        os.makedirs(output_data_dir, exist_ok=True)

        # TODO Remember to seed the numpy and python random generators, prior to every model running.
        # Iterate through all of the random seeds provided.
        for j, seed in enumerate(random_seeds):
            if print_progress:
                print('\t%d / %d; Current random seed: %d;' % (j+1, len(random_seeds), seed))

            # Models for any task-type
            if 'CATD' in models:
                pass

            if 'PM' in models:
                pass

            if dataset.task_type == 'regression':
                if 'mean' in models:
                    pass

                if 'median' in models:
                    pass

                # Most likley, all non-probablistic methods can be saved together

                # Truth Inference Survey 2017
                if 'LFC-N' in models:
                    pass

            elif 'classification' in dataset.task_type:
                if 'binary' in dataset.task_type:
                    # Use binary classifiaction only TI models
                    if 'Multi' in models:
                        pass

                    if 'KOS' in models:
                        pass

                    if 'VI-BP' in models:
                        pass

                    if 'VI-MF' in models:
                        pass

                # Use the multi-classification TI models

                # An Evaluation of Aggregation Technique in Crowdsourcing 2013
                if 'majority_vote' in models:
                    pass

                if 'majority_decision' in models:
                    pass

                if 'honeypot' in models:
                    pass

                if 'ELICE' in models:
                    pass

                if 'ipeirotis_dawid_skene' in models:
                    pass

                if 'SLME' in models:
                    pass

                if 'ITER' in models:
                    pass

                # Multi-Class Ground Truth Inference in Crowdsourcing with Clustering 2016
                if 'spectral_dawid_skene' in models:
                    pass

                if 'GTIC' in models:
                    pass

                # Truth Inference Survey 2017
                if 'dawid_skene' in models:
                    # for parameters in models['dawid_skene']: #if list of params.
                    # NOTE, EM is given a prior initial quality! if none, set all to 0.5, or random chance that the annotator is quality (ie. 1/#labels)

                    # Create the filepath to this instance's directory
                    dir_path = os.path.join(output_data_dir, 'dawid_skene', str(seed), '_'.join([key + '-' + str(value) for key, value in models['dawid_skene'].items()]))
                    # Make the  directory structure if it does not already exist.
                    os.makedirs(dir_path, exist_ok=True)

                    dawid_skene(samples_to_annotators, annotators_to_samples, dataset.label_set, models['dawid_skene'], dir_path, dataset_id, dataset_filepath, seed)

                if 'ZenCrowd' in models:
                    pass

                if 'GLAD' in models:
                    # Create the filepath to this instance's directory
                    dir_path = os.path.join(output_data_dir, 'GLAD', str(seed), '_'.join([key + '-' + str(value) for key, value in models['GLAD'].items()]))
                    # Make the  directory structure if it does not already exist.
                    os.makedirs(dir_path, exist_ok=True)

                    glad(samples_to_annotators, annotators_to_samples, dataset.label_set, models['GLAD'], dir_path, dataset_id, dataset_filepath, seed)

                if 'minimax' in models:
                    pass

                if 'BCC' in models:
                    pass

                if 'CBCC' in models:
                    pass

                if 'LFC' in models:
                    pass

                # Comparison of Bayesian Models of Annotation 2018
                if 'multinomial' in models:
                    pass

                if 'hier_dawid_skene' in models:
                    pass

                if 'item_diff' in models:
                    pass

                if 'log_rnd_eff' in models:
                    pass

                if 'MACE' in models:
                    pass


def dawid_skene(samples_to_annotators, annotators_to_samples, label_set, model_parameters, output_dir, dataset_id, dataset_filepath, random_seed):
    """Calls the Zheng 2017 implementation of Dawid and Skene EM algorithm and
    saves the results and runtimes of the method.
    """
    # Record start times
    datetime_start = datetime.now()
    start_process_time = process_time()
    start_performance_time = perf_counter()

    # Run the expectation maximization method.
    sample_label_probabilities, worker_confusion_matrices, = zheng_2017.EM(samples_to_annotators, annotators_to_samples, label_set, model_parameters['prior_quality']).Run(model_parameters['max_iterations'])

    # Record end times
    end_process_time = process_time()
    end_performance_time = perf_counter()
    datetime_end = datetime.now()

    # e2lpd: example to likelihood probability distribution
    # w2cm: workers to confusion matrix

    # Save the results.
    # Unpack the confusion matrix
    worker_confusion_matrix = worker_confusion_matrices.popitem()
    cm_df = pd.DataFrame(worker_confusion_matrix[1])
    number_of_label_values = len(worker_confusion_matrix[1])
    cm_df['worker_id'] = [worker_confusion_matrix[0]] * number_of_label_values

    if len(worker_confusion_matrices) > 0:
        for worker in worker_confusion_matrices:
            cm = pd.DataFrame(worker_confusion_matrices[worker])
            cm['worker_id'] = [worker] * len(worker_confusion_matrices[worker])
            cm_df = cm_df.append(cm)

    # Need to make worker_id the index/reorder, and save the label values.
    cm_df['label_value'] = cm_df.index
    cm_df = cm_df[['worker_id', 'label_value'] + list(cm_df['label_value'].iloc[:number_of_label_values])]

    # Save csv.
    cm_df.to_csv(os.path.join(output_dir, 'annotator_label_value_confusion_matrix.csv'), index=False)

    # Unpack the sample label probability estimates
    sample_label_probabilities = pd.DataFrame(sample_label_probabilities).T
    sample_label_probabilities.index.name = 'sample_id'
    sample_label_probabilities.to_csv(os.path.join(output_dir, 'sample_label_probabilities.csv'))

    # Create summary.csv
    summary_csv(os.path.join(output_dir, 'summary.csv'), 'dawid_skene', model_parameters, dataset_id, dataset_filepath, random_seed, end_process_time-start_process_time, end_performance_time-start_performance_time, datetime_start, datetime_end)


def glad(samples_to_annotators, annotators_to_samples, label_set, model_parameters, output_dir, dataset_id, dataset_filepath, random_seed):
    """Calls the Zheng 2017 implementation of GLAD and saves the results and
    runtimes of the method.
    """
    # Record start times
    datetime_start = datetime.now()
    start_process_time = process_time()
    start_performance_time = perf_counter()

    # Run the expectation maximization method.
    sample_label_probabilities, weight = zheng_2017.GLAD(samples_to_annotators, annotators_to_samples, label_set).Run(model_parameters['threshold'])

    # Record end times
    end_process_time = process_time()
    end_performance_time = perf_counter()
    datetime_end = datetime.now()

    # e2lpd: example to likelihood probability distribution
    # weight : a single weight for every annotator. worker_id : float

    # Save the results.
    # Unpack the sample label probability estimates
    sample_label_probabilities = pd.DataFrame(sample_label_probabilities).T
    sample_label_probabilities.index.name = 'sample_id'
    sample_label_probabilities.to_csv(os.path.join(output_dir, 'sample_label_probabilities.csv'))

    # Save weights as a csv
    weight = pd.DataFrame(weight, index=['weight']).T
    weight.index.name = 'worker_id'
    weight.to_csv(os.path.join(output_dir, 'weight.csv'))

    # Create summary.csv
    summary_csv(os.path.join(output_dir, 'summary.csv'), 'GLAD', model_parameters, dataset_id, dataset_filepath, random_seed, end_process_time-start_process_time, end_performance_time-start_performance_time, datetime_start, datetime_end)


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
    if args.random_seeds is None:
        args.random_seeds = config['random_seeds_file']
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
    parser.add_argument('-r',  '--random_seeds', default=None, help='The filepath to the file that contains the random seeds to use for the tests.')
    parser.add_argument('-i', '--iterations', default=None, type=int, help='The number of iterations to run the set of models for on the data and also the number of random seeds that will be generated in output file. This is ignored if random_seeds is provided via arguments or the configuration file.')

    parser.add_argument('--silent', action='store_true', help='Providing this flag disables all progress print outs from the script.')

    args =  parser.parse_args()

    # TODO Ensure all arguments are valid and load those missing from config
    if args.config is None and (args.datasets is None or args.models is None or args.output_dir is None):
        raise Exception('Must provide a configuration file via the config flag or pass all arguments when calling this script.')
    elif args.config is not None:
        # Load the values from the config file into args that are not present in args
        args = load_config(args)

    # Confirm output destination is valid and create if does not exist.
    if args.output_dir is None:
        raise Exception('No output directory provided.')
    elif os.path.isfile(args.output_dir):
        raise Exception('`%s` is a file, not a directory.' % args.output_dir)
    else:
        # attempt to create the output directory
        if args.overwrite_output_dir:
            warnings.warn('The `overwrite_output_dir` argument has been passed. The provided output directory exists and this program may overwrite existing files if the generated results share the same filename.', UserWarning)

        # if already exists then raise an error, unless overwrite_output_dir
        os.makedirs(args.output_dir, exist_ok=args.overwrite_output_dir)

    # Ensure the random seeds were provided, or create them
    if isinstance(args.random_seeds, str) and os.path.isfile(args.random_seeds):
        # Random seeds file exists, load the random seeds from file.
        with open(args.random_seeds, 'r') as random_seeds_file:
            args.random_seeds = []
            for seed in random_seeds_file:
                args.random_seeds.append(int(seed))

    elif args.random_seeds is None and isinstance(args.iterations, int):
        warnings.warn('A random seeds file was not provided. The random seeds wll be generate for every iteration, totaling  % (iter)d random seeds. A file containing these seeds will be saved along with the output at ` % (output)s/random_seeds_count- % (iter)d.txt`.' % {'iter':args.iterations, 'output':args.output_dir}, UserWarning)
        args.random_seeds = random_seed_generator.generate(args.iterations)

        # Save the newly generated random seeds to a file:
        random_seed_generator.save_to_file(args.random_seeds, os.path.join(args.output_dir, 'random_seeds_count-%d.txt' % args.iterations))

    elif args.random_seeds is None and not isinstance(args.iterations, int):
        raise Exception('The random seeds file was not provided and nor was the number of desired iterations. The random seeds were unable to be generated. Please provide either a random seeds file or a number of iterations.')

    # Confirm datasets are valid
    if args.datasets is None:
        raise Exception('No datasets provided.')

    unrecognized_datasets = set()
    for dataset in args.datasets:
        if not data_handler.dataset_exists(dataset):
            warnings.warn('Unrecognized dataset `%s`. This dataset will be ignored' % dataset, UserWarning)
            unrecognized_datasets.add(dataset)

            # Remove the dataset from the datasets_filepaths dict, if present
            if isinstance(args.datasets_filepaths, dict):
                args.datasets_filepaths.pop(dataset, None)

    # Remove unrecognized datasets
    args.datasets = set(args.datasets) - unrecognized_datasets
    if len(args.datasets) <= 0:
        raise Exception('There are no remaining datasets after removing the unrecognized datasets.')

    # TODO Confirm that the truth inference models are valid.
    if args.models is None:
        raise Exception('No models provided.')

    # TODO check that the models is a dict of parameters(dict) (or dict of list of dict of parameters)
    #unrecognized_models = set()
    #for model in args.models:
    #    if not truth_inference_model_handler.model_exists(model):
    #        warnings.warn('Unrecognized model `%s`. This model will be ignored'  % model, UserWarning)
    #        unrecognized_models.add(model)

    # Remove unrecognized truth inference models
    #args.models = set(args.models) - unrecognized_models
    #if len(args.models) <= 0:
    #    raise Exception('There are no remaining models after removing the unrecognized models.')

    return args


if __name__ == '__main__':
    # Read in arguements and configuaration file
    args = parse_args()

    # TODO Run the experiments,
    run_experiments(args.datasets, args.models, args.output_dir, args.random_seeds, args.datasets_filepaths, not args.silent)
