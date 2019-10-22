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

    def __init__(target, predictor_output, data_type='discrete'):
        """
        Parameters
        ----------
        target : np.ndarray | tfp.distribution.Distribution
            The target data of the supervised learning task.
        predictor_output : np.ndarray
            The predictions of the predictor for the samples corresponding to
            the target data.
        data_type : str
            Identifier of the data type of the target and the
        """
        assert(target.shape == predictor_output.shape)

        # Get transform matrix from data
        self.transform_matrix = self._get_change_of_basis_matrix(
            target.shape[1]
        )

        # TODO parallelize the fitting of the target and predictor distribs
        # Fit the target data
        self.target_distribution

        # Fit the transformation function of the target to the predictor output
        self.transform_distribution

    def _get_change_of_basis_matrix(self, input_dim):
        """Returns an invertable square matrix that transforms from an `input_dim`
        space representation of a discrete probability simplex to a `input_dim` - 1
        dimension probability simplex.

        Parameters
        ----------
        input_dim : int
            The dimensions of the space of the data of a discrete probability
            distribution. The valid values in this `input-dim`-space can be
            expressed as a probability simplex where all points in that space can
            be expressed as vectors whose values sum to 1 and are all in the range
            [0,1].

        Returns
        -------
        np.ndarray
            Invertable matrix that is used to convert from the `input_dim`-space
            to the probabilistic simplex space as a left-transfrom matrix and
            converts the other way as a right-transform matrix.
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
