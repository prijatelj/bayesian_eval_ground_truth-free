"""Uses the Supervised Joint Distribution to analyze the specified human data
and the predictors trained on that data.
"""
import logging
import os

import numpy as np

from psych_metric.distrib.supervised_joint_distrib import SupervisedJointDistrib

from experiment import io
import experiment.distrib
from experiment.research.sjd import src_candidates
from experiment.research.measure import measure
from experiment.research.bnn.bnn_mcmc_fwd import load_bnn_io_json


def mult_candidates_exp1(
    candidates,
    train,
    test=None,
    sample_size=100,
    normalize=False,
    ):
    """
    Runs multiple instances of experiment 1 with the given set of candidates
    and data

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
        target and the second is the predictions of a predictor. This optional
        and by default is assumed as not given.
    """
    # iterate through the candidate SJDs to obtain their results
    results = {}
    for key, kws in candidates.items():
        if isinstance(kws, SupervisedJointDistrib):
            candidate = kws
        else:
            # fit appropriate SJDs to train data.
            candidate = SupervisedJointDistrib(
                target=train[0],
                pred=train[1],
                **kws,
            )

        # Save parameters
        results[key] = {'params': candidate.params}

        # Get log prob exp results on in-sample data:
        results[key]['train'] = exp1_givens_data(
            candidate,
            train[0],
            train[1],
            sample_size,
            normalize=normalize,
        )

        if test is not None:
            # Get out of sample log prob exp results
            results[key]['test'] = exp1_givens_data(
                candidate,
                test[0],
                test[1],
                sample_size,
                normalize,
            )

    return results


def exp1_givens_data(
    candidate,
    targets,
    preds,
    sample_size=100,
    normalize=False,
):
    """Calculates the euclidean distance of the candidate
    SupervisedJointDistrib models' conditional distrib samples to the actual
    prediction value using the actual data of the givens. This is to test ONLY
    the conditional distrib, not the entire SupervisedJointDistrib (excluding
    the distrib of the givens).

    Parameters
    ----------
    candidates : SupervisedJointDistrib
        instance of a SupervisedJointDistrib whose log prob is to be estimated.
    target : np.ndarray
        Used as input to the SJD to generate the conditoinal distribution to be
        evaluated against the actual predicitons via Euclidean distance
        (residuals)
    pred : np.ndarray
    """
    # TODO sample from given candidate sjd, the conditional distrib only.
    if candidate.independent:
        conditional_samples = candidate.sample(
            len(targets) * sample_size,
        )[1].reshape(len(targets), sample_size, targets.shape[1])
    else:
        conditional_samples = np.stack(
            [
                candidate.transform_distrib.sample(targets)
                for i in range(sample_size)
            ],
            axis=2,
        ).reshape(len(targets), sample_size, targets.shape[1])

    return measure.get_l2dists(preds, conditional_samples, normalize)


def add_custom_args(parser):
    # just adding a this script specific argument
    measure.add_custom_args(parser)

    parser.add_argument(
        '--src_candidates',
        default=None,
        nargs='+',
        type=str,
        help='The premade candidate SJDs to be tested.'
    )

    parser.add_argument(
        '--test_datapath',
        default=None,
        type=str,
        help='The filepath to the JSON containing the test input and outputs.'
    )

    parser.add_argument(
        '--sample_size',
        default=100,
        type=int,
        help=' '.join([
            'The number of samples to draw from the conditional distribution',
            'per given sample.'
        ])
    )


if __name__ == '__main__':
    args = experiment.io.parse_args(['mle', 'sjd'], add_custom_args)

    output_dir = io.create_dirs(args.output_dir)

    # NOTE be aware that the defaults of SJD args will overwrite src candidates
    del args.sjd.target_distrib
    del args.sjd.transform_distrib
    del args.sjd.independent
    del args.sjd.mle_args

    logging.info('Loading data')
    train = load_bnn_io_json(args.data.dataset_filepath)

    if args.test_datapath is not None:
        test = load_bnn_io_json(args.data.dataset_filepath)
    else:
        test = None

    if args.src_candidates is None:
        args.src_candidates = [
            'iid_uniform_dirs',
            'iid_dirs_mean',
            #'iid_dirs_adam',
            'dir-mean_mvn-umvu',
            #'dir-adam_mvn-umvu',
        ]

    logging.info('Getting candidates')
    # Get candidates
    candidates = src_candidates.get_sjd_candidates(
        args.src_candidates,
        train[0].shape[1],
        vars(args.mle),
        vars(args.sjd),
        n_jobs=args.cpu_cores,
    )

    # Loop through candidates, fit givens and conds, sample conds given
    # data givens,
    if test is not None:
        results = mult_candidates_exp1(
            candidates,
             train,
             test,
             normalize=args.normalize,
         )
    else:
        results = mult_candidates_exp1(
            candidates,
             train,
             normalize=args.normalize,
         )

    # Save the results of multiple candidates.
    for candidate in args.src_candidates:
        tmp_out_dir = io.create_dirs(
            os.path.join(output_dir, candidate, 'train'),
        )
        measure.save_measures(
            tmp_out_dir,
            'euclid_dists_train',
            results[candidate]['train'],
            args.quantiles_frac,
            args.do_not_save_raw,
        )

        if test is not None:
            tmp_out_dir = io.create_dirs(
                os.path.join(output_dir, candidate, 'test'),
            )
            measure.save_measures(
                tmp_out_dir,
                'euclid_dists_test',
                results[candidate]['test'],
                args.quantiles_frac,
                args.do_not_save_raw,
            )
