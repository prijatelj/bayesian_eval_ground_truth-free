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
        self,
        target,
        pred,
        target_distrib,
        transform_distrib,
        data_type='nominal',
        independent=False,
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
        independent : bool, optional
            The joint probability distribution's random variables are treated
            as independent if True, otherwise the second random variable is
            treated as dependent on the first (this is default). This affects
            how the sampling is handled, specifically in how the
            `transform_distrib` is treated. If True, the `transform_distrib` is
            instead treated as the distrib of second random variable.
        """
        if target.shape != pred.shape:
            raise ValueError('`target.shape` and `pred.shape` must be the '
                + f'same. Instead recieved shapes {target.shape} and '
                + f'{pred.shape}.')

        self.independent = independent

        # Get transform matrix from data
        self.transform_matrix = self._get_change_of_basis_matrix(
            target.shape[1]
        )

        # TODO parallelize the fitting of the target and predictor distribs

        # Fit the target data
        if isinstance(target_distrib, tfp.distributions.Distribution):
            # Use given distribution as the fitted distribution.
            self.target_distrib = target_distrib
        elif isinstance(target_distrib, dict):
            # If given a dict, fit the distrib to the data
            if target_distrib['distrib_id'] == 'DirichletMultinomial':
                self.target_distrib = distribution_tests.mle_adam(**target_distrib)
            else:
                raise ValueError('Currently only "DirichletMultinomial" for '
                + 'the target distribution is supported as proof of concept.')
        else:
            raise TypeError('`target_distrib` is expected to be either of type '
                + '`tfp.distributions.Distribution` or `dict`.')


        # Fit the transformation function of the target to the predictor output
        if isinstance(transform_distrib, tfp.distributions.Distribution):
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
        """Creates a matrix that transforms from an `input_dim` space
        representation of a `input_dim` - 1 discrete probability simplex to
        that probability simplex's space.

        The first dimension is arbitrarily chosen to collapse and is the
        expected dimension to be adjusted for when moving the origin in the
        change of basis.

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
        # Create n-1 spanning vectors, Using first dim as origin of new basis.
        spanning_vectors = np.vstack((
            -np.ones(input_dim -1),
            np.eye(input_dim -1),
        ))

        # Create orthonormal basis of simplex
        return np.linalg.qr(spanning_vectors)[0].T

    def _transform_to(self, sample):
        """Transforms the sample from discrete distribtuion space into the
        probability simplex space of one dimension less. The orgin adjustment
        is used to move to the correct origin of the probability simplex space.
        """
        origin_adjust = np.zeros(len(self.transform_matrix))
        origin_adjust[0] = 1
        return self.transform_matrix @ (sample - origin_adjust)

    def _transform_from(self, sample):
        """Transforms the sample from probability simplex space into the
        discrete distribtuion space of one dimension more. The orgin adjustment
        is used to move to the correct origin of the discrete distribtuion space.
        """
        # NOTE tensroflow optimization instead here.
        origin_adjust = np.zeros(len(self.transform_matrix) + 1)
        origin_adjust[0] = 1
        return (sample @ self.transform_matrix) + origin_adjust

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
        target_samples = self.target_distrib.sample(num_samples)

        if self.independent:
            # just sample from transform_distrib and return paired RVs.
            transform_samples = self.transform_distrib.sample(num_samples)
            samples =  tf.stack([target_samples, transform_samples], 1)

            return samples.eval(session=tf.Session())

        # draw predictor output via transform function using target sample
        # pred_sample = target_sample + sample_transform_distrib
        transform_samples = self.transform_distrib.sample(num_samples)
        # Add the target sample to the transform sample to undo distance calc
        samples = tf.stack(
            [target_samples, transform_samples + target_samples],
            1,
        )
        joint_samples = samples.eval(session=tf.Session())

        # transform predictor output samples back to original space
        # NOTE this would probably be more optimal if simply saved the graph
        # for sampling and doing post transform in Tensorflow.
        for i in range(samples):
            joint_samples[i, 1] = self._transform_from(samples[i, 1])

        #return samples.eval(session=tf.Session())
        return joint_samples

    def _fit_transform_distrib(self, target, pred, distrib, distrib_id='normal'):
        """Fits and returns the transform distribution."""

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

        # NOTE logically, mean should be zero given the simplex and distances.
        # Will need to resample more given

        # deterministically calculate the Maximum Likely multivatiate norrmal
        return tfp.distributions.MultivariateNormalFullCovariance(
            #np.zeros(pred.shape[1]),
            np.mean(pred.shape[1], axis=0),
            np.cov(distances, bias=False, rowvar=False),
        )
