"""The general Input / Output of experiments."""
import argparse
from copy import deepcopy
import json
import logging
import os

import numpy as np
import keras
import tensorflow as tf


class NestedNamespace(argparse.Namespace):
    """An extension of the Namespace allowing for nesting of namespaces.

    Notes
    -----
    Modified version of hpaulj's answer at
        https://stackoverflow.com/a/18709860/6557057

    Use by specifying the full `dest` parameter when adding the arg. then pass
    the NestedNamespace as the `namespace` to be used by `parser.parse_args()`.
    """
    def __setattr__(self, name, value):
        if '.' in name:
            group, _, name = name.partition('.')
            ns = getattr(self, group, NestedNamespace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value

    def __getattr__(self, name):
        if '.' in name:
            group, _, name = name.partition('.')

            try:
                ns = self.__dict__[group]
            except KeyError:
                raise AttributeError

            return getattr(ns, name)
        else:
            getattr(super(NestedNamespace, self), name)


def save_json(filepath, results, additional_info=None, deep_copy=True):
    """Saves the content in results and additional info to a JSON.

    Parameters
    ----------
    filepath : str
        The filepath to the resulting JSON.
    results : dict
        The dictionary to be saved to file as a JSON.
    additional_info : dict
        Additional information to be added to results (to be removed).
    deep_copy : bool
        Deep copies the dictionary prior to saving due to making the contents
        JSON serializable.
    """
    if deep_copy:
        results = deepcopy(results)
    if additional_info:
        # TODO remove this if deemed superfulous
        results.update(additional_info)

    with open(filepath, 'w') as summary_file:
        for key in results:
            if isinstance(results[key], np.ndarray):
                # save to numpy specifc dir and save as csv.
                results[key] = results[key].tolist()
            elif isinstance(results[key], np.integer):
                results[key] = int(results[key])
            elif isinstance(results[key], np.floating):
                results[key] = float(results[key])

        json.dump(results, summary_file, indent=4, sort_keys=True)


def add_mle_args(parser):
    mle = parser.add_argument_group('mle', 'Arguments pertaining to the '
        + 'Maximum Likelihood Estimation.')
    mle.add_argument(
        '--max_iter',
        default=10000,
        type=int,
        help='The maximum number of iterations for finding MLE.',
        dest='mle.max_iter',
    )

    mle.add_argument(
        '--num_top_likelihoods',
        default=1,
        type=int,
        help='The number of top MLEs to be saved for each distribution.',
        dest='mle.num_top_likelihoods',
    )

    # Tolerances
    mle.add_argument(
        '--tol_param',
        default=1e-8,
        type=float,
        help='The threshold of parameter difference to the prior parameters '
            + 'set before declaring convergence and terminating the MLE '
            + 'search.',
        dest='mle.tol_param',
    )
    mle.add_argument(
        '--tol_loss',
        default=1e-8,
        type=float,
        help='The threshold of difference to the prior negative log likelihood '
            + 'set before declaring convergence and terminating the MLE '
            + 'search.',
        dest='mle.tol_loss',
    )
    mle.add_argument(
        '--tol_grad',
        default=1e-8,
        type=float,
        help='The threshold of difference to the prior gradient set before '
            + 'declaring convergence and terminating the MLE search.',
        dest='mle.tol_grad',
    )

    # optimizer_args
    mle.add_argument(
        '--learning_rate',
        default=1e-3,
        type=float,
        help='A Tensor or a floating point vlaue. The learning rate.',
        dest='mle.optimizer_args.learning_rate',
    )
    mle.add_argument(
        '--beta1',
        default=0.9,
        type=float,
        help='A float value or a constant float tensor. The exponential decay '
            + 'rate for the 1st moment estimates.',
        dest='mle.optimizer_args.beta1',
    )
    mle.add_argument(
        '--beta2',
        default=0.999,
        type=float,
        help='A float value or a constant float tensor. The exponential decay '
            + 'rate for the 2nd moment estimates',
        dest='mle.optimizer_args.beta2',
    )
    mle.add_argument(
        '--epsilon',
        default=1e-08,
        type=float,
        help='A small constant for numerical stability. This epsilon is '
            + '"epsilon hat" in the Kingma and Ba paper (in the formula just '
            + 'before Section 2.1), not the epsilon in Algorithm 1 of the '
            + 'paper.',
        dest='mle.optimizer_args.epsilon',
    )
    mle.add_argument(
        '--use_locking',
        action='store_true',
        help='Use locks for update operations.',
        dest='mle.optimizer_args.use_locking',
    )

    # tb_summary_dir ?? handled by output dir? or summary dir


def expand_mle_optimizer_args(args):
    """Put all mle-related args in a single dictionary."""
    if args.mle.optimizer_args and isinstance(args.mle.optimizer_args, NestedNamespace):
        args.mle.optimizer_args = vars(args.mle.optimizer_args)
    elif args.mle.optimizer_args:
        raise TypeError(f'`args.mle.optimizer` has expected type NestedNamespace, but instead was of type {type(args.mle.optimizer)} with value: {args.mle.optimizer}')


def parse_args(arg_set=None):
    """Creates the args to be parsed and the handling for each.

    Parameters
    ----------
    arg_set : iterable, optional
        contains the additional argument types to be parsed.

    Returns
    -------
    (argparse.namespace, dict, dict, dict, None|int|list(ints))
        Args parsed, dicts of data, model, kfold specific args, and list of
        random seeds.
    """
    parser = argparse.ArgumentParser(description='Run proof of concept ')

    # TODO consider useing arg_set to specify the types of args to load.

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
        help=(
            'The focus fold to split the data on to form train and test sets '
            + 'for a singlemodel train and evaluate session (No K-fold Cross '
            + 'Validation; Just evaluates one partition).',
        )
    )

    # Output args
    parser.add_argument(
        '-o',
        '--output_dir',
        default='./',
        help='Filepath to the output directory.',
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

    if arg_set and 'mle' in arg_set:
        add_mle_args(parser)

    args = parser.parse_args(namespace=NestedNamespace())

    # Set logging configuration
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
    if args.log_file is not None:
        dir_part = args.log_file.rpartition(os.path.sep)[0]
        os.makedirs(dir_part, exist_ok=True)
        logging.basicConfig(filename=args.log_file, level=numeric_level)
    else:
        logging.basicConfig(level=numeric_level)

    # Set the Hardware
    keras.backend.set_session(tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=args.cpu_cores,
        inter_op_parallelism_threads=args.cpu_cores,
        allow_soft_placement=True,
        device_count={
            'CPU': args.cpu,
            'GPU': args.gpu,
        } if args.gpu >= 0 else {'CPU': args.cpu},
    )))

    # package the arguements:
    # TODO handle all packaging via NestedNamespaces in argparse, vars() later
    data_args = {'dataset_filepath': args.dataset_filepath}

    model_args = {
        'model_id': args.model_id,
        'init': {'crowd_layer': args.crowd_layer, 'kl_div': args.kl_div},
        'train': {'epochs': args.epochs, 'batch_size': args.batch_size},
        'parts': args.parts,
    }

    kfold_cv_args = {
        'kfolds': args.kfolds,
        'save_pred': args.no_save_pred,
        'save_model': args.no_save_model,
        'stratified': args.stratified,
        'test_focus_fold': not args.train_focus_fold,
        'shuffle': args.no_shuffle_data,
        # 'repeat': None,
        'period': args.period,
        'period_save_pred': args.period_save_pred,
    }

    if args.random_seeds and len(args.random_seeds) == 1:
        kfold_cv_args['random_seed'] = args.random_seeds[0]
        random_seeds = None
    else:
        random_seeds = args.random_seeds


    if arg_set and 'mle' in arg_set:
        expand_mle_optimizer_args(args)

    # TODO combine packaged args into same args namespace

    return args, data_args, model_args, kfold_cv_args, random_seeds
