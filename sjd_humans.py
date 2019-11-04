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
from experiment.kfold import kfold_generator, get_kfold
from psych_metric.supervised_joint_distrib import SupervisedJointDistrib

def load_summary(summary_file):
    """Recreates the variables for the experiment from the summary JSON file."""
    with open(summary_file, 'r'_ as fp:
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
    test_focus_fold : bool, optional
        If True, then expects that in kfold_cv_args, instead of
        `train_focus_fold` it willbe the inverse, `test_focus_fold` as the key.
        This is for support of an older version of the code. By default this is
        False.

    Returns
    -------
        the trained model
        the data
    """
    # Load summary json & obtain experiment variables for loading main objects.
    dataset_id, data_args, model_args, kfold_cv_args = load_summary(
        os.path.join(dir_oath, summary_name)
    )

    # load the data, if necessary
    if data is None:
        # Only load data if need to run to get predictions
        data, labels, label_bin = predictors.load_prep_data(
            dataset_id,
            data_config,
            label_src,
            model_config['parts'],
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
        data,
        # only pass labels if stratified, fine w/o for now
        shuffle=kfold_cv_args['shuffle'],
        stratified=kfold_cv_args['stratified'],
        train_focus_fold=train_focus_fold,
        random_seed=kfold_cv_args['random_seed'],
    )

    # Recereate the model
    model = predictors.load_model(
        model_config['model_id'],
        model_config['crowd_layer'],
        model_config['parts'],
        os.path.join(dir_path, weights_file),
        **model_config['init'],
    )

    # TODO perhaps return something to indicate #folds or the focusfold #
    #return model, data[train_idx], data[test_idx]
    return model, features[test_idx], labels[test_idx]


def sjd_kfold_exp(
    dir_path,
    weights_file,
    label_src,
    summary_name='summary.json',
    data=None,
):
    """Performs SJD test on kfold experiment by loading the model predictions
    form each fold and averages the results, saves error bars, and other
    distribution information useful from kfold cross validation.
    """

    # load data if given dict

    for dir_p in os.listdir(dir_path):
        model, features, labels = load_eval_fold(dir_p, data)

        # TODO generate predictions
        pred = model.predict(features)

        # fit SJD to train data.
        sjd = SupervisedJointDistrib(labels, )

        # perform Log Prob test on test/eval set.
        log_prob_results = test_human_sjd(sjd)

        # save em results

    # TODO aggregate the results, ie. average

    return results


def multiple_sjd_kfold_exp():
    """Performs SJD test on multiple kfold experiments and returns the
    aggregated result information.
    """

    return results

def add_sjd_human_test_args(parser):
    # TODO add the str id for the target to compare to: 'frequency', 'ground_truth'
    # Can add other annotator aggregation methods as necessary, ie. D&S.

if __name__ == '__main__':
    args, random_seeds = experiment.io.parse_args('mle', 'sjd')

    # TODO first, load in entirety a single eval fold

    # TODO then, recursively load many and save the SJD general results (ie. mean log_prob comparisons with error bars).
