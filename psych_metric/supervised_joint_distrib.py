"""Bayesian Joint Probability of target distribution and predictor output
distribution. Assumes that the predictor output is dependent on the target
distribution and that the prediction output can be obtained with some error by
a transformation function of the target distribution.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric import distribution_tests


class SupervisedJointDistrib(object):
    """Bayesian distribution fitting of a joint probability distribution whose
    values correspond to the supervised learning context with target data and
    predictor output data.

    Attributes
    ----------
    transformation_matrix : np.ndarray
        The matrix that transforms from the
    target_distribution : tfp.distribution.Distribution
        The probability distribution of the target data.
    transform_distribution : tfp.distribution.Distribution
        The probability distribution of the transformation function for
        transforming the target distribution to the predictor output data.
    """

    def __init__(
        target,
        pred,
        target_distrib,
        transform_distrib,
        data_type='nominal',
    ):
        """
        Parameters
        ----------
        target : np.ndarray
            The target data of the supervised learning task.
        pred : np.ndarray
            The predictions of the predictor for the samples corresponding to
            the target data.
        target_distrib : dict | tfp.distribution.Distribution
            Either the parameters to a  distribution to be used as the fitted
            distribution the target data, or the actual distribution to be
            used.
        transform_distrib : dict | tfp.distribution.Distribution
            Either the parameters to a  distribution to be used as the fitted
            distribution the target data, or the actual distribution to be
            used.
        data_type : str, optional
            Identifier of the data type of the target and the predictor output.
            Values include 'nominal', 'ordinal', and 'continuous'. Defaults to
            'nominal'.
        """
        if target.shape != pred.shape:
            raise ValueError('`target.shape` and `pred.shape` must be the '
                + f'same. Instead recieved shapes {taget.shape} and '
                + f'{pred.shape}.')

        # Get transform matrix from data
        self.transform_matrix = self._get_change_of_basis_matrix(
            target.shape[1]
        )

        # TODO parallelize the fitting of the target and predictor distribs

        # Fit the target data
        if isinstance(target_distrib, tfp.distribution.Distribution):
            # Use given distribution as the fitted distribution.
            self.target_distrib = target_distrib
        elif isinstance(target_distrib, dict):
            # If given a dict, fit the distrib to the data
            if target_distrib['distrib_id'] = 'DirichletMultinomial':
                self.target_distrib = distribution_tests.mle_adam(**target_distrib)
            else:
                raise ValueError('Currently only "DirichletMultinomial" for '
                + 'the target distribution is supported as proof of concept.')
        else:
            raise TypeError('`target_distrib` is expected to be either of type '
                + '`tfp.distributions.Distribution` or `dict`.')


        # Fit the transformation function of the target to the predictor output
        if isinstance(transform_distrib, tfp.distribution.Distribution):
            self.transform_distrib = transform_distrib
        elif isinstance(transform_distrib, dict):
            self.transform_distrib = self._fit_transform_distrib(
                target,
                pred,
                transform_distrib,
            )
        else:
            raise TypeError('`transform_distrib` is expected to be either of '
                + 'type `tfp.distributions.Distribution` or `dict`.')

    def _get_change_of_basis_matrix(self, input_dim):
        """Returns an invertable square matrix that transforms from an
        `input_dim` space representation of a discrete probability simplex to a
        `input_dim` - 1 dimension probability simplex.

        Parameters
        ----------
        input_dim : int
            The dimensions of the space of the data of a discrete probability
            distribution. The valid values in this `input-dim`-space can be
            expressed as a probability simplex where all points in that space
            can be expressed as vectors whose values sum to 1 and are all in
            the range [0,1].

        Returns
        -------
        np.ndarray
            Invertable matrix that is used to convert from the
            `input_dim`-space to the probabilistic simplex space as a
            left-transfrom matrix and converts the other way as a
            right-transform matrix.
        """

        # TODO obtain the transform matrix (change of basis matrix)

        return

    def _transform_to(self, sample):
        return

    def _transform_from(self, sample):
        return

    def sample(self, num_samples):
        """Sample from the estimated joint probability distribution.

        Returns
        -------
        np.ndarray, shape(samples, input_dim, 2)
            Array of samples from the joint probability distribution of the
            target and the predictor output.
        """
        # sample from target distribution

        # transform target samples to probability simplex

        # draw predictor output via transform function using target sample

        # transform predictor output samples back to original space

        return joint_prob_samples

    def _fit_transform_distrib(self, target, pred, distrib, distrib_id='normal'):
        """Fits the transform distribution."""

        # TODO make some handling of MLE not converging if using adam, ie. to
        # allow the usage of non-convergence mle or not. (should be fine using
        # multivariate gaussian)
        if not isinstance(distrib, dict):
            # NOTE may be unnecessary given expectation to only be called by class code
            raise TypeError('`distrib` is expected to be of type `dict` not '
                + f'`{type(distrib)}`')

        if distrib_id != 'normal' or distrib_id != 'gaussian':
            raise ValueError('Currently only "gaussian" or "normal" is '
            + 'supported for the transform distribution as proof of concept.')

        distances = self._transform_to(pred) - self._transform_to(target)

        # mean = 0, estimate covariances from distances
        locs = np.zeros(pred.shape[1])
        scales = np.std(pred, 0)
        #covs = np.cov(pred)

        # TODO deterministically calculate the multivatiate gaussian distrib

        return transform_distrib
