"""Code the tests and verifies the functionality and validity of the fitting of
the supervised joint distribution with dependency between the joint random
variables.
"""
import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import experiment.io
import experiment.distrib
from experiment.kfold import kfold_generator
from psych_metric import distribution_tests
from psych_metric.supervised_joint_distrib import SupervisedJointDistrib


def test_identical(
    output_dir,
    concentration,
    num_classes,
    loc,
    covariance_matrix,
    kfolds=5,
    sample_size=1000,
    info_criterions=['bic', 'aic', 'hqc'],
    random_seed=None,
    test_independent=True,
    mle_args=None,
):
    """Creates a Dirichlet-multinomial distribution for the target's source and
    a multivariate normal distribution for the transformation function, then
    fits those using the SupervisedJointDistrib class to ensure it is able to
    fit that. This is the most straight-forward and basic test.

    Parameters
    ----------
    output_dir : str
        The filepath to the directory where the MLE results will be stored.
    mle_args : dict
        Arguments for MLE Adam.
    total_count : float | dict, optional
        either a float as the initial value of the loc, or a dict containing
        the loc and standard deviation of a normal distribution which this
        loc is drawn from randomly.
    concentration : float | list(float), optional
        either a postive float as the initial value of the scale, or a dict
        containing the loc and standard deviation of a normal distribution
        which this loc is drawn from randomly. If
    num_classes : int
        The number of classes of the source Dirichlet-Multinomial. Only
        required when the given a single float for `concentration`.
        `concentration` is then turned into a list of length `num_classes`
        where ever element is the single float given by `concentration`.
    sample_size : int, optional
        The number of samples to draw from the normal distribution being fit.
        Defaults to 1000.
    info_criterions : list(str)
        List of information criterion ids to be computed after finding the MLE.
    random_seed : int, optional
        Integer to use as the random seeds for MLE.
    """
    # Create the source target distribution
    src_target_params = experiment.distrib.get_dirichlet_params(
        concentration,
        num_classes,
    )

    # Create the source transform distribution
    src_transform_params = experiment.distrib.get_multivariate_normal_full_cov_params(
        loc,
        covariance_matrix,
        num_classes,
    )

    # Combine them into a joint distribution
    src_joint_distrib = SupervisedJointDistrib(
        tfp.distributions.Dirichlet(**src_target_params),
        tfp.distributions.MultivariateNormalFullCovariance(
            **src_transform_params
        ),
        sample_dim=len(src_target_params['concentration']),
    )

    # Create the sample data that the distribs will fit.
    target, pred = src_joint_distrib.sample(sample_size)

    # Save the distribs args
    distrib_args = {'source_distrib': {
        'target_distrib': src_target_params,
        'transform_distrib': src_transform_params,
    }}

    # Principle of Indifference distribution (the lowest baseline) No dependence
    distrib_args['principle_of_indifference'] = {
        'target_distrib': {
            'concentration': [1] * target.shape[1],
        },
    }
    distrib_args['principle_of_indifference']['transform_distrib'] = distrib_args['principle_of_indifference']['target_distrib']

    principle_of_indifference = SupervisedJointDistrib(
        tfp.distributions.Dirichlet(
            **distrib_args['principle_of_indifference']['target_distrib'],
        ),
        tfp.distributions.Dirichlet(
            **distrib_args['principle_of_indifference']['transform_distrib'],
        ),
        sample_dim=target.shape[1],
        independent=True,
    )

    # Dict to store all info and results for this test as a JSON.
    results = {
        'kfolds': kfolds,
        'invariant_distribs': distrib_args,
        'focus_folds': {},
    }

    # number of params = 2 * |concentration|
    num_independent_params = 2 * target.shape[1]
    # number of params = |concentration| + |class means| + |covariances|
    num_src_params = num_independent_params + target.shape[1] * (target.shape[1] + 1) / 2

    # Create list of distribs to loop through in each fold.
    distribs = ['src', 'poi', 'umvu']
    if test_independent:
        distribs += ['independent_umvu']
    if mle_args is not None:
        distribs += ['umvu_mle']
        if test_independent:
            distribs += ['independent_umvu_mle']

    # K Fold CV of fitting methods UMVUE and MLE (via Adam) SJDs.
    for i, (train_idx, test_idx) in enumerate(kfold_generator(kfolds, target)):
        if random_seed:
            np.random.seed(random_seed)
            tf.set_random_seed(random_seed)

        focus_fold = {}

        for distrib in distribs:
            if distrib == 'src':
                sjd = src_joint_distrib
                num_params = num_src_params
            elif distrib == 'poi':
                sjd = principle_of_indifference
                num_params = num_independent_params
            elif distrib == 'umvu':
                sjd = SupervisedJointDistrib(
                    'Dirichlet',
                    'MultivariateNormal',
                    target[train_idx],
                    pred[train_idx],
                )
                num_params = num_src_params
            elif distrib == 'umvu_mle':
                sjd = SupervisedJointDistrib(
                    'Dirichlet',
                    'MultivariateNormal',
                    target[train_idx],
                    pred[train_idx],
                    mle_args=mle_args,
                )
                num_params = num_src_params
            elif distrib == 'independent_umvu':
                sjd = SupervisedJointDistrib(
                    'Dirichlet',
                    'Dirichlet',
                    target[train_idx],
                    pred[train_idx],
                    independent=True,
                )
                num_params = num_independent_params
            elif distrib == 'independent_umvu_mle':
                sjd = SupervisedJointDistrib(
                    'Dirichlet',
                    'Dirichlet',
                    target[train_idx],
                    pred[train_idx],
                    mle_args=mle_args,
                    independent=True,
                )
                num_params = num_independent_params

            # In sample log prob
            focus_fold[distrib]['train']['log_prob'] = sjd.log_prob(
                target[train_idx],
                pred[train_idx],
            )
            # In sample info criterions
            focus_fold[distrib]['train']['info_criterion'] = distribution_tests.calc_info_criterion(
                focus_fold[distrib]['train']['log_prob'],
                num_params,
                info_criterions,
                num_samples=len(train_idx),
            )

            # Out sample log prob
            focus_fold[distrib]['test']['log_prob'] = sjd.log_prob(
                target[test_idx],
                pred[test_idx],
            )
            # Out sample info criterions
            focus_fold[distrib]['test']['info_criterion'] = distribution_tests.calc_info_criterion(
                focus_fold[distrib]['test']['log_prob'],
                num_params,
                info_criterions,
                num_samples=len(test_idx),
            )

        # Save all the results for this fold.
        results['focus_folds'][i + 1] = focus_fold

    # TODO Iteratively save ?  Save the results
    experiment.io.save_json(
        os.path.join(output_dir, 'test_SJD_identical.json'),
        results,
    )


def test_identity_transform():
    """This is the basic test of SupervisedJointDistrib that tests how the
    multivariate normal transform handles the identity as the transform
    function. This is useful to compare how the SupervisedJointDistrib with
    dependent joint random variables compares to the independent random
    variables.

    The dependent case should still perform better because even with
    the exact same distribution, the random variables could be out of sync with
    each otherand thus not have identical draws, assuming they are not given
    the same random seed (which of course means they would then match
    completely).
    """
    pass


def test_arg_extreme(argmax=True):
    """Tests the SupervisedJointDistrib on how it handles when the transform
    function is either argmax or argmin, where the extremum of the discrete
    probabilities is set to 1 and the rest are set 0.
    """
    pass


def test_shift(shift=1, shift_right=True):
    """Tests the SupervisedJointDistrib on how it handles when the transform
    function is shifts the probabilities of the classes in one direction or the
    other for so many spots with wrap around.
    """
    pass


def add_test_sjd_args(parser):
    """Adds the test SJD arguments to the argparser."""
    sjd = parser.add_argument_group(
        'sjd',
        'Arguments pertaining to tests evaluating the'
        + 'SupervisedJointDistribution in fitting simulated data.',
    )

    sjd.add_argument(
        '--test_id',
        default='identical',
        help='The SupervisedJointDistrib test to be performed.',
        choices=[
            'identical',
            'identity_transform',
            'arg_extreme',
        ],
        dest='sjd.test_id',
    )

    sjd.add_argument(
        '--sample_size',
        default=1000,
        type=int,
        help='The number of samples to draw from the source distribution.',
        dest='sjd.sample_size',
    )

    # concentration
    sjd.add_argument(
        '--concentration',
        type=experiment.io.multi_typed_arg(
            float,
            lambda x: np.array(x.split(), dtype=float),
        ),
        help=' '.join([
            'Either a positive float or a list of postive floats indicating',
            'the concentrations for as many classes there are in ',
        ]),
        dest='sjd.concentration',
        required=experiment.io.check_argv(
            ['test_identical'],
            '--test_id',
        ),
    )

    sjd.add_argument(
        '--num_classes',
        type=int,
        help='Number of classes in the data.',
        dest='sjd.num_classes',
        required=(
            experiment.io.check_argv(
                ['test_identical'],
                '--test_id',
            )
            and experiment.io.check_argv(float, '--concentration')
        ),
    )


    # loc
    sjd.add_argument(
        '--loc',
        type=experiment.io.multi_typed_arg(
            float,
            lambda x: np.array(x.split(), dtype=float),
        ),
        help=' '.join([
            'Either a positive float or a list of postive floats indicating',
            'the means for each dimension of the Multivariate Normal. Default',
            'value of None results in a randomly selected value for the loc.',
        ]),
        default=None,
        dest='sjd.loc',
    )

    # covariance_matrix: random,
    sjd.add_argument(
        '--covariance_matrix',
        type=float,
        help=' '.join([
            'A positive float for the value of all diagonal values in the',
            'covariance matrix. Default value of None results in a randomly',
            'selected covariance matrix.',
        ]),
        default=None,
        dest='sjd.covariance_matrix',
    )


    # sjd.knn fitting size


if __name__ == '__main__':
    # TODO write up argparse code for this (possibly repurposing older args).
    args, random_seeds = experiment.io.parse_args(
        'mle',
        add_test_sjd_args,
        description=' '.join([
            'Perform tests via simulated data to evaluate the quality of',
            'SupervisedJointDistribution in fitting arbitrary data.'
        ]),
    )

    #if args.sjd.test_id == 'identical':
    test_identical(
        args.output_dir,
        args.sjd.concentration,
        args.sjd.num_classes if args.sjd.num_classes else len(args.sjd.concentration),
        args.sjd.loc,
        args.sjd.covariance_matrix,
        args.kfold_cv.kfolds,
        args.sjd.sample_size,
        mle_args=vars(args.mle),
    )
