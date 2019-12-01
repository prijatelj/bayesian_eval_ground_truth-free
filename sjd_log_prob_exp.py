"""Uses the Supervised Joint Distribution to analyze the specified human data
and the predictors trained on that data.
"""
import csv
import json
import logging
import os

import h5py
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distrib import distrib_utils
from psych_metric.supervised_joint_distrib import SupervisedJointDistrib

import experiment.io
import experiment.distrib
from experiment.kfold import kfold_generator, get_kfold_idx
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
    raise NotImplementedError


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
        (labels[train_idx], labels[test_idx]),
        #kfold_cv_args['focus_fold'],
    )


def old_mean_results(log_prob_results, info_criterions=None):
    """Average the results"""
    # Obtain mean log prob for both datasets and all distribs
    results = {}
    for candidate in log_prob_results[0].keys():
        results[candidate] = {}

        for dataset in {'train', 'test'}:
            results[candidate][dataset] = {'log_prob': {}}

            if info_criterions:
                results[candidate][dataset]['info_criterion'] = {}

            for distrib in {'joint', 'target', 'transform'}:
                # mean log prob
                results[candidate][dataset]['log_prob'][distrib] = np.mean([
                    fold[candidate][dataset]['log_prob'][distrib]
                    for fold in log_prob_results
                ])

                # mean info criterions
                if not info_criterions:
                    continue
                results[candidate][dataset]['info_criterion'][distrib] = {}

                for ic in info_criterions:
                    results[candidate][dataset]['info_criterion'][distrib][ic] = np.mean([
                        fold[candidate][dataset]['info_criterion'][distrib][ic]
                        for fold in log_prob_results
                    ])

    return results


def mean_results(log_prob_results, info_criterions=None):
    """Average the results"""
    # TODO make an updated mean_results, this is for the older format.
    raise NotImplementedError

    # Obtain mean log prob for both datasets and all distribs
    results = {}
    for candidate in log_prob_results[0].keys():
        results[candidate] = {}

        for dataset in {'train', 'test'}:
            results[candidate][dataset] = {'log_prob': {}}

            if info_criterions:
                results[candidate][dataset]['info_criterion'] = {}

            for distrib in {'joint', 'target', 'transform'}:
                # mean log prob
                results[candidate][dataset]['log_prob'][distrib] = np.mean([
                    fold[candidate][dataset]['log_prob'][distrib]
                    for fold in log_prob_results
                ])

                # mean info criterions
                if not info_criterions:
                    continue
                results[candidate][dataset]['info_criterion'][distrib] = {}

                for ic in info_criterions:
                    results[candidate][dataset]['info_criterion'][distrib][ic] = np.mean([
                        fold[candidate][dataset]['info_criterion'][distrib][ic]
                        for fold in log_prob_results
                    ])

    return results


def sjd_kfold_log_prob(
    candidates,
    dir_path,
    weights_file,
    label_src,
    summary_name='summary.json',
    data=None,
    load_model=True,
    info_criterions=None,
    output_path=None,
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
    # Use data if given tuple, otherwise load each time given dict
    if isinstance(data, dict):
        data = predictors.load_prep_data(**data)

    results = []
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

        # TODO generate predictions XOR if already given preds, use them
        if load_model:
            pred = (model.predict(features[0]), model.predict(features[1]))
        else:
            pred = model
            del model

        # Get Eval fold's results
        results.append(log_prob_exps(
            candidates,
            (labels[0], pred[0]),
            (labels[1], pred[1]),
            info_criterions,
        ))

    if output_path:
        # Save the results to file
        experiment.io.save_json(output_path, results)

    return results


def multiple_sjd_kfold_log_prob(
    dir_paths,
    sjd_args,
    dir_path,
    weights_file,
    label_src,
    summary_name='summary.json',
    data=None,
    load_model=True,
    info_criterions=['bic', 'aic', 'hqc'],
):
    """Performs SJD test on multiple kfold experiments and returns the
    aggregated result information.
    """
    # Recursively walk a root directory and get the applicable directories,
    # OR be given the filepaths to the different kfold experiments.

    log_prob_results = []
    for dir_p in dir_paths:
        if not os.path.isdir(dir_p):
            logging.warning('Not a directory: skipping %s', dir_p)
            continue

        log_prob_results.append(mean_results(sjd_kfold_log_prob(
            sjd_args,
            dir_p,
            weights_file,
            label_src,
            summary_name,
            data,
            load_model,
            info_criterions=info_criterions,
        )))

    return log_prob_results


def src_log_prob_exp(
    src,
    candidates,
    info_criterions=None,
    num_samples=1000,
    json_path=None,
    calc_src=True,
    src_id='src',
):
    """Compares the log probability of the candidate SupervisedJointDistrib
    models to the given source distribution in a simulated test.

    Parameters
    ----------
    src : SupervisedJoinDistrib
    candidates : dict(str: SupervisedJointDistrib)
    num_samples : int, optional
    json_path : str, optional
        Writes the results to a JSON file located at this filepath if given.
    calc_src : bool, optional
        If True, calculates the log prob experiment results of the src
        distribution fitting itself. This is True by default as it provides the
        upper bound of performance.
    src_id : str, optional
        the identifier used in the results dictionary for the source
        distribution.

    Returns
    -------
    dict
        A dictionary whose keys are the identifiers of the candidates and
        source SupervisedJointDistrib that points to a dictionary containing
        the parameters and results.
    """
    # Sample the data from the src to be used to assess the candidate SJDs
    train = src.sample(num_samples)
    test = src.sample(num_samples)

    if calc_src:
        # Add the src to the candidates to calculate its results
        candidates[src_id] = src

    results = log_prob_exps(candidates, train, test, info_criterions)

    if json_path:
        # Save the results to file
        experiment.io.save_json(json_path, results)

    return results


def log_prob_exps(
    candidates,
    train,
    test,
    info_criterions=None,
):
    """
    Runs multiple log prob experiments with the given set of candidates and
    data

    Parameters
    ----------
    candidates : dict
        Dictionary of str candidate identifiers keys to values of either.
    train : tuple(np.ndarray, np.ndarray)
        Data used for fitting the SJD and testing the typical Bayesian
        situation. Tuple of same shaped ndarrays where the first is the target
        and the second is the predictions of a predictor.
    test : tuple(np.ndarray, np.ndarray)
        Data used for testing the fitted SJD only in a cross validation
        scenerio, different from the typical Bayesian situation of testing
        fitted models. Tuple of same shaped ndarrays where the first is the
        target and the second is the predictions of a predictor.
    info_criterions : list(str)
        List of str identifiers of which information criterions to calculate
        after calculating the log probability.
    """
    # iterate through the candidate SJDs to obtain their results
    results = {}
    for key, kws in candidates.items():
        if not isinstance(kws, SupervisedJointDistrib):
            candidate = kws
        else:
            # fit appropriate SJDs to train data.
            candidate = SupervisedJointDistrib(
                target=train[0],
                pred=train[1],
                **kws,
            )

        # Save parameters
        results[key] = {'params': distrib_utils.get_sjd_params(candidate)}

        # Get log prob exp results on in-sample data:
        results[key]['train'] = log_prob_exp(
            candidate,
            train[0],
            train[1],
            info_criterions,
        )

        # Get out of sample log prob exp results
        results[key]['test'] = log_prob_exp(
            candidate,
            test[0],
            test[1],
            info_criterions,
        )

    return results


def log_prob_exp(
    candidate,
    target,
    pred,
    num_params,
    info_criterions=None,
):
    """Calculates the log probability of the candidate SupervisedJointDistrib
    models.

    Parameters
    ----------
    candidates : dict(str: SupervisedJointDistrib)
        Dict containing the string identifier of each SJD model to their
        respective instance of a SupervisedJointDistrib.
    target : np.ndarray
    pred : np.ndarray
    info_criterions : list(str), optional
    """
    # Calculate the log probability (log likelihood)
    log_probs = candidate.log_prob(target, pred)
    results = {var:{'log_prob': log_probs[i]} for i, var in
        enumerate(['joint', 'target', 'transform'])
    }

    if info_criterions:
        # Calculate any information criterions (target, pred, joint)
        for var, value in results:
            # Get appropriate number of parameters from the SJD
            if var == 'target':
                num_params = candidate.target_num_params
            elif var == 'transform':
                num_params = candidate.transform_num_params
            elif var == 'joint':
                if value['log_prob'] is None:
                    # Joint was not able to be calculated.
                    continue
                num_params = candidate.num_params

            value['info_criterion'] = distrib_utils.calc_info_criterion(
                value['log_prob'],
                info_criterions,
                num_params,
                len(target),
            )

    return results


def log_prob_test_human_sjd(
    fit_sjd,
    target,
    pred,
    sjd_args,
    info_criterions=['bic','aic','hqc'],
    distribs=['uniform', 'independent_umvu', 'independent_umvu_mle', 'umvu', 'fit_sjd'],
    tf_sess_config=None,
):
    """Compares the log probability of the fitted SupervisedJointDistrib to
    other baselines.

    """
    log_probs = {}

    # Dict to store all info and results for this test as a JSON.
    results = {distrib: {'train':{}, 'test':{}} for distrib in distribs}
    results['uniform']['params'] = {
        'target': {'concentration': [1] * target[0].shape[1]},
        'transform': {'concentration': [1] * target[0].shape[1]},
    }

    # Concentration is number of classes
    num_dir_params = target[0].shape[1]
    # Mean is number of classes, and Covariance Matrix is a triangle matrix
    num_mvn_params = target[0].shape[1] + target[0].shape[1] * (target[0].shape[1] + 1) / 2

    num_params_dependent = {
        'joint': num_dir_params + num_mvn_params,
        'target': num_dir_params,
        'transform': num_mvn_params,
    }
    num_params_independent = {
        'joint': 2 * num_dir_params,
        'target': num_dir_params,
        'transform': num_dir_params,
    }

    for distrib in distribs:
        # Create each distrib being tested.
        if distrib == 'uniform':
            sjd = SupervisedJointDistrib(
                tfp.distributions.Dirichlet(
                    **results['uniform']['params']['target'],
                ),
                tfp.distributions.Dirichlet(
                    **results['uniform']['params']['transform'],
                ),
                sample_dim=target[0].shape[1],
                independent=True,
                tf_sess_config=tf_sess_config,
            )
            num_params = num_params_independent
        elif 'independent' in distrib:
            sjd = SupervisedJointDistrib(
                'Dirichlet',
                'Dirichlet',
                target[0],
                pred[0],
                mle_args=None if distrib == 'independent_umvu' else sjd_args['mle_args'],
                independent=True,
                tf_sess_config=tf_sess_config,
            )

            num_params = num_params_independent

            results[distrib]['params'] = {
                'target':{
                    'concentration': sjd.target_distrib._parameters['concentration'].tolist()
                },
                'transform':{
                    'concentration': sjd.transform_distrib._parameters['concentration'].tolist()
                }
            }
        else:
            if distrib == 'umvu':
                sjd = SupervisedJointDistrib(
                    'Dirichlet',
                    'MultivariateNormal',
                    target[0],
                    pred[0],
                    tf_sess_config=tf_sess_config,
                )
            else:
                sjd = fit_sjd

            num_params = num_params_independent

            results[distrib]['params'] = {
                'target':{
                    'concentration': sjd.target_distrib._parameters['concentration'].tolist()
                },
                'transform':{
                    'loc': sjd.transform_distrib._parameters['loc'].tolist(),
                    'covariance_matrix': sjd.transform_distrib._parameters['covariance_matrix'].tolist()
                }
            }

        # calculate log prob of all sjds
        # In sample log prob
        results[distrib]['train']['log_prob'] = sjd.log_prob(
            target[0],
            pred[0],
            return_individuals=True,
        )
        results[distrib]['train']['log_prob'] = {
            'joint': results[distrib]['train']['log_prob'][0].sum(),
            'target': results[distrib]['train']['log_prob'][1].sum(),
            'transform': results[distrib]['train']['log_prob'][2].sum(),
        }
        # In sample info criterions
        info_crit = {}
        for rv, log_prob in results[distrib]['train']['log_prob'].items():
            info_crit[rv] = distrib_utils.calc_info_criterion(
                log_prob,
                num_params[rv],
                info_criterions,
                num_samples=len(target[0]),
            )
        results[distrib]['train']['info_criterion'] = info_crit

        # Out sample log prob
        results[distrib]['test']['log_prob'] = sjd.log_prob(
            target[1],
            pred[1],
            return_individuals=True,
        )
        results[distrib]['test']['log_prob'] = {
            'joint': results[distrib]['test']['log_prob'][0].sum(),
            'target': results[distrib]['test']['log_prob'][1].sum(),
            'transform': results[distrib]['test']['log_prob'][2].sum(),
        }
        # Out sample info criterions
        info_crit = {}
        for rv, log_prob in results[distrib]['test']['log_prob'].items():
            info_crit[rv] = distrib_utils.calc_info_criterion(
                log_prob,
                num_params[rv],
                info_criterions,
                num_samples=len(target[1]),
            )
        results[distrib]['test']['info_criterion'] = info_crit

    return results


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
    args = experiment.io.parse_args(
        ['mle', 'sjd'],
        add_human_sjd_args,
    )

    # TODO first, load in entirety a single eval fold
    """
    uh = sjd_kfold_log_prob(
        #sjd_args=vars(args.sjd),
        dir_path=args.human_sjd.dir_path,
        weights_file=args.human_sjd.weights_file,
        label_src=args.label_src,
        summary_name=args.human_sjd.summary_name,
        data=None,
        load_model=True,
        #info_criterions=args.,
    )
    """
    # TODO then try with load data ONCE, load one summary of a kfold. use data for all.
    #data=args.human_sjd.dir_path,

    # TODO then do a single kfold experiment

    # TODO then, recursively load many and save the SJD general results (ie. mean log_prob comparisons with error bars).
