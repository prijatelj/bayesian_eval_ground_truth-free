"""Simulation of a distribution in sampled data and fitting with different
distributions.
"""
import logging

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distrib.supervised_joint_distrib import SupervisedJointDistrib

import experiment.distrib

def get_src_sjd(sjd_id, dims, sjd_args=None):
    """Returns SupervisedJointDistrib of hardcoded simulation source
    distributions.
    """
    if sjd_id == 'iid_uniform_dirs':
        # two independent Dirichlets whose concentrations are all ones
        sjd_kws = {
            'target_distrib': tfp.distributions.Dirichlet(np.ones(dims)),
            'transform_distrib': tfp.distributions.Dirichlet(np.ones(dims)),
            'independent': True,
            'sample_dim': dims,
        }

        if sjd_args:
            sjd_kws.update(sjd_args)

        return SupervisedJointDistrib(**sjd_kws)

    if sjd_id == 'random_dir_mvn':
        # Random Dirichlet with random MVN transformation
        sjd_kws = {
            'target_distrib': tfp.distributions.Dirichlet(
                **experiment.distrib.get_dirichlet_params(num_classes=dims),
            ),
            'transform_distrib': tfp.distributions.MultivariateNormalFullCovariance(
                **experiment.distrib.get_multivariate_normal_full_cov_params(
                    sample_dim=dims - 1,
                ),
            ),
            'independent': False,
            'sample_dim': dims,
        }

        if sjd_args:
            sjd_kws.update(sjd_args)

        return SupervisedJointDistrib(**sjd_kws)

    if sjd_id == 'uniform_dir_small_mvn':
        # Random Dirichlet with random small MVN transformation. so the
        # identity transform with some noise.
        sjd_kws = {
            'target_distrib': tfp.distributions.Dirichlet(
                **experiment.distrib.get_dirichlet_params(np.ones(dims)),
            ),
            'transform_distrib': tfp.distributions.MultivariateNormalFullCovariance(
                **experiment.distrib.get_multivariate_normal_full_cov_params(
                    loc=0.0,
                    covariance_matrix=np.eye(dims - 1) * 1e-4,
                    sample_dim=dims - 1,
                ),
            ),
            'independent': False,
            'sample_dim': dims,
        }

        if sjd_args:
            sjd_kws.update(sjd_args)

        return SupervisedJointDistrib(**sjd_kws)

    if sjd_id == 'tight_dir_small_mvn':
        # Random Dirichlet with random small MVN transformation. so the
        # identity transform with some noise over a dirichlet with a precision
        sjd_kws = {
            'target_distrib': tfp.distributions.Dirichlet(
                **experiment.distrib.get_dirichlet_params(np.ones(dims) * 10),
            ),
            'transform_distrib': tfp.distributions.MultivariateNormalFullCovariance(
                **experiment.distrib.get_multivariate_normal_full_cov_params(
                    loc=0.0,
                    covariance_matrix=np.eye(dims - 1) * 1e-4,
                    sample_dim=dims - 1,
                ),
            ),
            'independent': False,
            'sample_dim': dims,
        }

        if sjd_args:
            sjd_kws.update(sjd_args)

        return SupervisedJointDistrib(**sjd_kws)


def get_sjd_candidates(
    sjd_ids,
    dims,
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
    dims : int
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
        candidates['iid_uniform_dirs'] = get_src_sjd('iid_uniform_dirs', dims)
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

        # TODO may need to allow passing of different MLE args to the separate
        # distribs. ie Dir almost always uses Grad Descent, but MVC anything.

        candidates['dir_mvc_mle'] = {
            'target_distrib': 'Dirichlet',
            'transform_distrib': 'MultivariateNormal',
            'indpendent': False,
            'mle_args': mle_args_copy,
            'processes': processes,
        }

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
