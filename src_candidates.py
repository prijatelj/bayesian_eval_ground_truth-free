"""Simulation of a distribution in sampled data and fitting with different
distributions.
"""
import logging

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import experiment.distrib
import psych_metric.supervised_joint_distrib import SupervisedJointDistrib


def get_src_sjd(sjd_id, sjd_args=None):
    """Returns actual SupervisedJointDistrib."""
    if sjd_id == 'iid_uniform_dirs':
        # two independent Dirichlets whose concentrations are all ones
        sjd_kws = {
            'target_distrib': tfp.distirbutions.Dirichlet(np.ones(sample_dim)),
            'transform_distrib': tfp.distirbutions.Dirichlet(
                np.ones(sample_dim)
            ),
            'indpendent': True,
        }

        if sjd_args:
            sjd_kws.update(sjd_args)

        return SupervisedJointDistrib(**sjd_kws)


def get_sjd_candidates(
    sjd_id,
    sample_dim,
    mle_args=None,
    processes=16,
    sjd_args=None,
):
    """Creates the dictionary of candidate identifers to their
    SupervisedJointDistrib initialization arguments. This is a convenience
    function for creating common versions of the SJD for empirical comparison
    (ie. baselines and the recommended versions).

    Parameters
    ----------
    sjd_id : str | list
        The different types of SupervisedJointDistributions candidates to be
        created. Each id is associated with its own set of initialization
        arguments for SupervisedJointDistributions.
    sample_dim : int
        The number of dimensions of the sample space. This assumes a discrete
        sample space.
    mle_args : dict, optional
        the Maximum Likelihood Estimation args to be used when fitting the
        data.
    processes : int, optional
        The number of CPU processes to be used when performing estimation of
        parameters of the distributions.
    sjd_args : dict, optional
        dictionary of SupervisedJointDistrib initialization arguments that will
        be added to or override each candidate created in this function.
    """
    candidates = {}

    # NOTE the dicts exclude

    if 'iid_uniform_dirs' in sjd_ids:
        candidates['iid_uniform_dirs'] = get_src_sjd('iid_uniform_dirs')
    if 'iid_dirs_mean' in sjd_ids:
        # two independent Dirichlets whose concentrations are the means of data
        candidates['iid_dirs_mean'] = {
            'target_distrib': 'Dirichlet',
            'transform_distrib': 'Dirichlet',
            'indpendent': True,
        }
        if sjd_args:
            candidates['iid_dirs_mean'].update(sjd_args)
    if 'iid_dirs_mle' in sjd_ids:
        # two independent Dirichlets whose concentrations are the means of data
        # multiplied by the precisions found via MLE
        mle_args_copy = mle_args.copy()
        mle_args_copy['alt_distrib'] = True

        candidates['iid_dirs_mle'] = {
            'target_distrib': 'Dirichlet',
            'transform_distrib': 'Dirichlet',
            'indpendent': True,
            'mle_args': mle_args_copy,
            'processes': processes,
        }
        if sjd_args:
            candidates['iid_dirs_mle'].update(sjd_args)
    if 'dir-mean_mvn-umvu' in sjd_ids:
        # target: Dirichlet: concentration is mean of data
        # transform: Multivariate Normal: loc and cov matrix from data
        candidates['dir-mean_mvn-umvu'] = {
            'target_distrib': 'Dirichlet',
            'transform_distrib': 'MultivariateNormal',
            'indpendent': False,
            'mle_args': mle_args,
            'processes': processes,
        }
        if sjd_args:
            candidates['dir-mean_mvn-umvu'].update(sjd_args)
    if 'dir-mle_mvn-umvu' in sjd_ids:
        # target: Dirichlet: concentration is mean of data * mle precision
        # transform: Multivariate Normal: loc and cov matrix from data
        mle_args_copy = mle_args.copy()
        mle_args_copy['alt_distrib'] = True

        candidates['dir-mle_mvn-umvu'] = {
            'target_distrib': 'Dirichlet',
            'transform_distrib': 'MultivariateNormal',
            'indpendent': False,
            'mle_args': mle_args_copy,
            'processes': processes,
        }
        if sjd_args:
            candidates['dir-mle_mvn-umvu'].update(sjd_args)
    if 'dir_mvc_mle' in sjd_ids:
        # target: Dirichlet: concentration is mean of data * mle precision
        # transform: Multivariate Cauchy: loc and cov matrix initialized from
        # data and estimated via a MLE method... TODO
        mle_args_copy = mle_args.copy()
        mle_args_copy['alt_distrib'] = True


        if sjd_args:
            candidates['dir_mvc_mle'].update(sjd_args)
    if 'dir_mvst_mle' in sjd_ids:
        # target: Dirichlet: concentration is mean of data * mle precision
        # transform: Multivariate Student T: loc and cov matrix initialized
        # from data and estimated via a MLE method... TODO
        mle_args_copy = mle_args.copy()
        mle_args_copy['alt_distrib'] = True

        if sjd_args:
            candidates['dir_mvst_mle'].update(sjd_args)
    if 'dir-mle_bnn-euclid' in sjd_ids:
        # target: Dirichlet: concentration is mean of data * mle precision
        # transform: BNN in Euclidean space trained via Random Walk.
        raise NotImplementedError
        mle_args_copy = mle_args.copy()
        mle_args_copy['alt_distrib'] = True

        if sjd_args:
            candidates['dir-mle_bnn-euclid'].update(sjd_args)
    if 'dir-mle_bnn-hyperbolic-nuts' in sjd_ids:
        # target: Dirichlet: concentration is mean of data * mle precision
        # transform: BNN in Hyperbolic space trained via NUTS.
        raise NotImplementedError
        mle_args_copy = mle_args.copy()
        mle_args_copy['alt_distrib'] = True

        if sjd_args:
            candidates['dir-mle_bnn-hyperbolic-nuts'].update(sjd_args)
    # TODO opt. estimation methods: Nelder-Mead, Simmulated Annealing, MCMC,
    # Gradient descent
    # TODO optional hyperbolic transform or clipped euclidean w/ resampling
    # TODO constant zeros for location in transform for distances.

    return candidates
