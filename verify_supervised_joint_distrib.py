"""Code the tests and verifies the functionality and validity of the fitting of
the supervised joint distribution with dependency between the joint random
variables.
"""
import argparse
import json
import logging
import os
from numbers import Number
import sys

import numpy as np
from sklearn.neighbors.kde import KernelDensity
import tensorflow as tf
import tensorflow_probability as tfp

import experiment.io
import experiment.distrib
from experiment.kfold import kfold_generator
from psych_metric import distribution_tests
from psych_metric.supervised_joint_distrib import SupervisedJointDistrib


def test_identical(
    output_dir,
    #total_count,
    concentration,
    num_classes,
    sample_size=1000,
    info_criterions=['bic', 'aic', 'hqc'],
    random_seed=None,
    #init_total_count=None,
    init_concentration=None,
    repeat_mle=1,
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
        #total_count,
        concentration,
        num_classes,
    )

    # Create the source transform distribution
    src_transform_params =  experiment.distrib.get_multivariate_normal_full_cov_params(
        loc,
        covariance_matrix,
    )

    src_joint_distrib = SupervisedJointDistrib(
        tfp.distributions.Dirichlet(**src_target_params),
        tfp.distributions.MultivariateNormalFullCovariance(
            **src_transform_params
        ),
        sample_dim=len(src_target_params['concentration']),
    )

    data = src_joint_distrib.sample(sample_size)

    # TODO set the other distribs, POI, UMVUE, MLE_adam (if flag is True)
    distrib_args = {'source_distrib': {
        'target_distrib': src_target_params,
        'transform_distrib': src_transform_params,
    }}

    # Principle of Indifference distribution (the lowest baseline) No dependence
    distrib_args['principle_of_indifference'] = {
        'target_distrib': {
            #'total_count': src_target_params['total_count'],
            'concentration': [1] * data.shape[1],
        },
    }
    distrib_args['principle_of_indifference']['transform_distrib'] = distrib_args['principle_of_indifference']['target_distrib']

    principle_of_indifference = SupervisedJointDistrib(
        tfp.distributions.Dirichlet(
            **distrib_args['principle_of_indifference']['target_distrib'],
        ),
        tfp.distributions.MultivariateNormalFullCovariance(
            **distrib_args['principle_of_indifference']['transform_distrib'],
        ),
        sample_dim=data.shape[1],
        independent=True,
    )

    # Dict to store all info and results for this test as a JSON.
    results = {'kfolds': kfolds, 'invariant_distribs': distrib_args, 'focus_folds': {}}

    # concentration + class means + covariances
    num_src_params = data.shape[1] * 2 + np.ceil((data.shape[1] ** 2) / 2)

    # K Fold CV of fitting methods UMVUE and MLE (via Adam) SJDs.
    for i, (train_idx, test_idx) in enumerate(kfold_generator(kfolds, data)):
        if random_seed:
            np.random.seed(random_seed)
            tf.set_random_seed(random_seed)

        focus_fold = {}

        umvue = SupervisedJointDistrib(
            'Dirichlet',
            'MultivariateNormal',
            target[train_idx],
            pred[train_idx],
        )

        # evaluate on in & out of sample: SRC, POI, UMVUE, Independents
        # calculate Likelihood

        # in-sample/training log prob
        focus_fold['src']['train']['log_prob'] = src.log_prob(target[train_idx], pred[train_idx])
        focus_fold['poi']['train']['log_prob'] = poi.log_prob(target[train_idx], pred[train_idx])
        focus_fold['umvue']['train']['log_prob'] = umvue.log_prob(target[train_idx], pred[train_idx])

        # out-sample/testing log prob
        log_prob['test']['src'] = src.log_prob(target[test_idx], pred[test_idx])
        log_prob['test']['poi'] = poi.log_prob(target[test_idx], pred[test_idx])
        log_prob['test']['umvue'] = umvue.log_prob(target[test_idx], pred[test_idx])


        for distrib in distrib_args:
            # in-sample/training log prob
            log_prob['train']['src'] = src.log_prob(target[train_idx], pred[train_idx])

            # calculate Information criterions
            focus_fold['']= distribution_tests.calc_info_criterion(
                umvue_log_prob,
                num_src_params,
            )

            # out-sample/testing log prob
            log_prob['test']['src'] = src.log_prob(target[test_idx], pred[test_idx])


        if test_independent:
            independent_umvue = SupervisedJointDistrib(
                'Dirichlet',
                'Dirichlet',
                target[train_idx],
                pred[train_idx],
                independent=True,
            )

        # TODO If mle_args given, use an MLE method
        if mle_args is not None:
            umvue_mle = SupervisedJointDistrib(
                'Dirichlet',
                'MultivariateNormal',
                target[train_idx],
                pred[train_idx],
                mle_args=mle_args,
            )

            focus_fold['mle']['train']['log_prob']

        # Save all the results for this fold.
        results['focus_folds'][i + 1] = focus_fold



    # TODO Iteratively save ?  Save the results

    experiment.io.save_json(
        os.path.join(args.output_dir, 'test_SJD_identical.json'),
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

if __name__ == '__main__':
    # TODO write up argparse code for this (possibly repurposing older args).
    raise NotImplementedError
