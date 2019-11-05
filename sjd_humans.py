"""Uses the Supervised Joint Distribution to analyze the specified human data
and the predictors trained on that data.
"""
import csv
import json
import os

import h5py
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold, StratifiedKFold

import experiment.io
from experiment.kfold import kfold_generator, get_kfold_idx
from psych_metric.supervised_joint_distrib import SupervisedJointDistrib
import predictors

def load_summary(summary_file):
    """Recreates the variables for the experiment from the summary JSON file."""
    with open(summary_file, 'r') as fp:
        summary = json.load(fp)

        dataset_id = 'LabelMe' if 'LabelMe' in summary else 'FacialBeauty'

        return (
            dataset_id,
            summary[dataset_id],
            summary['model_config'],
            summary['kfold_cv_args'],
        )


def recreate_label_bin(class_order):
    """Recreate the label binarizer as specified in the summary, if
    necesary.
    """
    return


def load_eval_fold(
    dir_path,
    weights_file,
    label_src,
    summary_name='summary.json',
    data=None,
    load_model=True,
):
    """Uses the information contained within the summary file to recreate the
    predictor model and load the data with its kfold indices.

    Parameters
    ----------
    dir_path : str
        The filepath to the eval fold directory.
    weight_file : str
        The relative filepath from the eval fold directory to the weights file
        to be used to load the model.
    label_src : str, optional
        The label_src expected to be used by the loaded model. Must be provided
        as the summary does not contain this.
    summary_name : str
        The relative filepath from the eval fold directory to the summary JSON
        file.
    data : ?
        If provided, then the data (information) to be used for getting Kfold
        indices. This is to avoid having to reload and prep the data if it can
        be already loaded and prepped beforehand.
    load_model : bool, optional
        Currently a placeholder var indicating the possibilty of simply loading
        a predictions file instead of the model weights. Thus returning the
        predictions and the labels for the kfold split.

    Returns
    -------
    tuple
        The trained model and the label data split into train and test sets.
    """
    if not load_model:
        NotImplementedError('loading the predictions directly is not supported at the moment.')

    # Load summary json & obtain experiment variables for loading main objects.
    dataset_id, data_args, model_args, kfold_cv_args = load_summary(
        os.path.join(dir_path, summary_name)
    )

    # load the data, if necessary
    if isinstance(data, tuple):
        # Unpack the variables from the tuple
        features, labels, label_bin = data
    else:
        # Only load data if need to run to get predictions
        features, labels, label_bin = predictors.load_prep_data(
            dataset_id,
            data_args,
            label_src,
            model_args['parts'],
        )

    # Support for summaries written with older version of code
    if 'test_focus_fold' in kfold_cv_args:
        train_focus_fold = not kfold_cv_args['test_focus_fold']
    else:
        train_focus_fold = kfold_cv_args['train_focus_fold']

    # Create kfold indices
    train_idx, test_idx = get_kfold_idx(
        kfold_cv_args['focus_fold'],
        kfold_cv_args['kfolds'],
        features,
        # only pass labels if stratified, fine w/o for now
        shuffle=kfold_cv_args['shuffle'],
        stratified=kfold_cv_args['stratified'],
        train_focus_fold=train_focus_fold,
        random_seed=kfold_cv_args['random_seed'],
    )

    # Recereate the model
    model = predictors.load_model(
        model_args['model_id'],
        parts=model_args['parts'],
        weights_file=os.path.join(dir_path, weights_file),
        **model_args['init'],
    )

    # TODO perhaps return something to indicate #folds or the focusfold #
    return (
        model,
        (features[train_idx], features[test_idx]),
        (labels[train_idx], features[test_idx]),
    )


def sjd_kfold_log_prob(
    sjd_args,
    dir_path,
    weights_file,
    label_src,
    summary_name='summary.json',
    data=None,
    load_model=True,
):
    """Performs SJD log prob test on kfold experiment by loading the model
    predictions form each fold and averages the results, saves error bars, and
    other distribution information useful from kfold cross validation.

    This checks if the SJD is performing as expected on the human data.

    Parameters
    ----------
    sjd_args : dict
        The arguments to be used for fitting the SupervisedJointDistrib to the
        data.
    dir_path : str
        The filepath to the eval fold directory.
    weight_file : str
        The relative filepath from the eval fold directory to the weights file
        to be used to load the model.
    label_src : str
        The label_src expected to be used by the loaded model. Must be provided
        as the summary does not contain this.
    summary_name : str, optional
        The relative filepath from the eval fold directory to the summary JSON
        file.
    data : tuple, dict, optional
        If provided, then this is the to be used for getting Kfold
        indices. This is to avoid having to reload and prep the data if it can
        be already loaded and prepped beforehand.
    """
    # load data if given dict
    if isinstance(data, dict):
        data = predictors.load_prep_data(**data)

    # Use data if given tuple, otherwise load each time.

    log_prob_results = []
    for ls_item in os.listdir(dir_path):
        dir_p = os.path.join(dir_path, ls_item)

        # Skip file if not a directory
        if not os.path.isdir(dir_p):
            continue

        model, features, labels = load_eval_fold(
            dir_p,
            weights_file,
            label_src,
            summary_name,
            data=data,
            load_model=load_model,
        )

        # TODO generate predictions XOR if made it so already given preds, use them
        if load_model:
            train_pred = model.predict(features[0])
            test_pred = model.predict(features[1])
        else:
            pred = model
            del model

        # fit SJD to train data.
        sjd = SupervisedJointDistrib(
            target=labels[0],
            pred=train_pred,
            **sjd_args,
        )
        """
        # perform Log Prob test on test/eval set.
        log_prob_results.append(log_prob_test_human_sjd(
            sjd,
            labels[1],
            test_pred,
            sjd_args,
        ))
        """
        log_prob_results.append(sjd)

    return log_prob_results


def multiple_sjd_kfold_log_prob(
    dir_paths,
    sjd_args,
    dir_path,
    weights_file,
    label_src,
    summary_name='summary.json',
    data=None,
    load_model=True,
):
    """Performs SJD test on multiple kfold experiments and returns the
    aggregated result information.
    """
    # Recursively walk a root directory and get the applicable directories,
    # OR be given the filepaths to the different kfold experiments.

    log_prob_results = []
    for dir_p in dir_paths:
        if not os.path.isdir(dir_p):
            # TODO log this and skip
            continue
        log_prob_results.append(sjd_kfold_log_prob(
            sjd_args,
            dir_p,
            weights_file,
            label_src,
            summary_name,
            data,
            load_model,
        ))

    return log_prob_results


def sjd_metric_cred():
    """Get a single model trained on all data, and create many samples to
    obtain the credible interval of the predictor output, joint distribution,
    and metrics calculated on that joint distribution.
    """
    # Fit SJD to human & preds if not already available.

    # sample many times from that SJD (1e6)

    # optionally save those samples to file.

    # Calculate credible interval given alpha for Joint Distrib samples

    # compare how that corresponds to the actual data human & pred.

    # save all this and visualize it.

    # Also, able to quantify % of data points outside of credible interval

    raise NotImplementedError()


def log_prob_test_human_sjd(fit_sjd, target, pred, sjd_args):
    """Compares the log probability of the fitted SupervisedJointDistrib to
    other baselines.
    """
    log_probs = {}

    # TODO create baseline sjds

    # calculate log prob of all sjds

    return log_probs


def add_human_sjd_args(parser):
    # TODO add the str id for the target to compare to: 'frequency', 'ground_truth'
    # Can add other annotator aggregation methods as necessary, ie. D&S.
    human_sjd = parser.add_argument_group(
        'human_sjd',
        'Arguments pertaining to log prob tests evaluating the'
        + 'SupervisedJointDistribution in fitting human data.',
    )

    human_sjd.add_argument(
        '--dir_path',
        help=' '.join([
            'The filepath to the directory containing either the separate',
            'results directories of a kfold experiment, or the root directory',
            'containing all kfold experiments to be loaded.',
        ]),
        default='./',
        dest='human_sjd.dir_path',
    )

    human_sjd.add_argument(
        '--weights_file',
        help=' '.join([
            'The relative filepath from the eval fold directory to the',
            'model weights HDF5 file.',
        ]),
        default='weights.hdf5',
        dest='human_sjd.weights_file',
    )

    human_sjd.add_argument(
        '--summary_name',
        help=' '.join([
            'The relative filepath from the eval fold directory to the',
            'summary JSON file.',
        ]),
        default='summary.json',
        dest='human_sjd.summary_name',
    )


if __name__ == '__main__':
    args, random_seeds = experiment.io.parse_args(
        ['mle', 'sjd'],
        add_human_sjd_args,
    )

    # TODO first, load in entirety a single eval fold
    uh = sjd_kfold_log_prob(
        sjd_args=vars(args.sjd),
        dir_path=args.human_sjd.dir_path,
        weights_file=args.human_sjd.weights_file,
        label_src=args.label_src,
        summary_name=args.human_sjd.summary_name,
        #data=None,
        #load_model=True,
    )

    # TODO then try with load data ONCE, load one summary of a kfold. use data for all.
    data=args.human_sjd.dir_path,

    # TODO then do a single kfold experiment

    # TODO then, recursively load many and save the SJD general results (ie. mean log_prob comparisons with error bars).
