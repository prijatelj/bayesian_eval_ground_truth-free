"""Bayesian Joint Probability of target distribution and predictor output
distribution. Assumes that the predictor output is dependent on the target
distribution and that the prediction output can be obtained with some error by
a transformation function of the target distribution.
"""
from copy import deepcopy
import math

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric import distribution_tests

# TODO handle continuous distrib of continuous distrib, only discrete atm.

class SupervisedJointDistrib(object):
    """Bayesian distribution fitting of a joint probability distribution whose
    values correspond to the supervised learning context with target data and
    predictor output data.

    Attributes
    ----------
    independent : bool
        True if the random variables are indpendent of one anothers, False
        otherwise. Default is False.
    transformation_matrix : np.ndarray
        The matrix that transforms from the
    target_distribution : tfp.distribution.Distribution
        The probability distribution of the target data.
    transform_distribution : tfp.distribution.Distribution
        The probability distribution of the transformation function for
        transforming the target distribution to the predictor output data.
    tf_sample_sess: tf.Session
    tf_target_samples: tf.Tensor
    tf_pred_samples: tf.Tensor
    tf_num_samples: tf.placeholder
    """

    def __init__(
        self,
        target_distrib,
        transform_distrib,
        target=None,
        pred=None,
        data_type='nominal',
        independent=False,
        sample_dim=None,
        total_count=None,
        tf_sess_config=None,
    ):
        """
        Parameters
        ----------
        target_distrib : dict | tfp.distribution.Distribution
            Either the parameters to a  distribution to be used as the fitted
            distribution the target data, or the actual distribution to be
            used.
        transform_distrib : dict | tfp.distribution.Distribution
            Either the parameters to a  distribution to be used as the fitted
            distribution the target data, or the actual distribution to be
            used.
        target : np.ndarray, optional
            The target data of the supervised learning task.
        pred : np.ndarray, optional
            The predictions of the predictor for the samples corresponding to
            the target data.
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
        sample_dim : int, optional
            The number of dimensions of a single sample of both the target and
            predictor distribtutions. This is only required when `target` and
            `pred` are not provided.
        total_count : int
            Non-zero, positive integer of total count for the
            Dirichlet-multinomial target distribution.
        """
        self.independent = independent
        if not isinstance(total_count, int):
            raise TypeError(' '.join([
                '`total_count` of type `int` must be passed in order to use',
                'DirichletMultinomial distribution as the target',
                f'distribution, but recieved type {type(total_count)}',
                'instead.',
            ]))
        self.total_count = total_count

        if target is not None and pred is not None:
            if target.shape != pred.shape:
                raise ValueError(' '.join([
                    '`target.shape` and `pred.shape` must be the same.',
                    f'Insteadrecieved shapes {target.shape} and {pred.shape}.',
                ]))

            # Get transform matrix from data
            self.transform_matrix = self._get_change_of_basis_matrix(
                target.shape[1]
            )
        elif isinstance(sample_dim, int):
            self.transform_matrix = self._get_change_of_basis_matrix(sample_dim)
        else:
            TypeError(' '.join([
                '`target` and `pred` must be provided together, otherwise',
                '`sample_dim` must be given instead, along with',
                '`target_distrib` and `transform_distrib` given explicitly as',
                'an already defined distribution each.',
            ]))

        # TODO parallelize the fitting of the target and predictor distribs

        # Fit the target data
        self.target_distrib = self._fit_independent_distrib(
            target_distrib,
            target,
        )

        # Fit the transformation function of the target to the predictor output
        if self.independent:
            self.transform_distrib = self._fit_independent_distrib(
                transform_distrib,
                pred,
            )
        elif isinstance(transform_distrib, tfp.distributions.Distribution):
            # Use given distribution as the fitted dependent distribution
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

        # Create the Tensorflow session and ops for sampling
        self._create_sampling_attributes(tf_sess_config)

        if not self.independent:
            # Create the predictor output pdf estimate
            self._create_empirical_predictor_pdf()
        else:
            self.pred_distrib = None

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        for k, v in self.__dict__.items():
            if k == 'transform_distrib' or k == 'target_distrib':
                setattr(new, k, v.copy())
            else:
                setattr(new, k, deepcopy(v, memo))

        return new

    def _fit_independent_distrib(self, distrib, data=None):
        """Fits the given data with an independent distribution."""
        if isinstance(distrib, tfp.distributions.Distribution):
            # Use given distribution as the fitted distribution.
            return distrib
        elif isinstance(distrib, str):
            # By default, use UMVUE of params of given data. No MLE fitting.
            if distrib == 'DirichletMultinomial':
                #total_count = np.sum(data, axis=1).max(0)
                total_count = math.fsum(data[0])
                return  tfp.distributions.DirichletMultinomial(
                    total_count,
                    np.mean(data, axis=0), #/ total_count,
                )
            elif distrib == 'MultivariateNormal':
                return tfp.distributions.MultivariateNormalFullCovariance(
                    np.mean(data, axis=0),
                    np.cov(data, bias=False, rowvar=False),
                )
            else:
                raise ValueError(' '.join([
                    'Currently only "DirichletMultinomial" or',
                    '"MultivariateNormal" for ' 'independent distributions ',
                    'are supported as proof of concept.',
                ]))
        elif isinstance(distrib, dict):
            # If given a dict, use as initial parameters and fit with MLE
            if distrib['distrib_id'] == 'DirichletMultinomial' or distrib['distrib_id'] == 'MultivariateNormal':
                return  distribution_tests.mle_adam(
                    data=data,
                    **distrib,
                )
            else:
                raise ValueError(' '.join([
                    'Currently only "DirichletMultinomial" or',
                    '"MultivariateNormal" for ' 'independent distributions ',
                    'are supported as proof of concept.',
                ]))
        else:
            raise TypeError(' '.join([
                '`distrib` is expected to be either of type',
                '`tfp.distributions.Distribution`, `str`, or `dict`, not',
                f'{type(dict)}.',
                'If a `str`, then it is the name of the distribution.',
                'If `dict`, then it is the parameters of the distribution ',
                'with "distrib_id" as a key to indicate which distribution.',
            ]))

    def _create_sampling_attributes(self, sess_config=None):
        """Creates the Tensorflow session and ops for sampling."""
        #raise NotImplementedError()

        self.tf_sample_sess = tf.Session(config=sess_config)

        num_samples = tf.placeholder(tf.int32, name='num_samples')

        self.tf_target_samples = self.target_distrib.sample(num_samples)

        if self.independent:
            # just sample from transform_distrib and return paired RVs.
            self.tf_pred_samples = self.transform_distrib.sample(num_samples)
        else:
            # Normalize the target samples for use with dependent transform
            norm_target_samples = self.tf_target_samples / self.total_count

            # Create the origin adjustment for going to and from n-1 simplex.
            origin_adjust = np.zeros(self.transform_matrix.shape[1])
            origin_adjust[0] = 1

            # Convert the normalized target samples into the n-1 simplex basis.
            #target_simplex_samples = self.transform_matrix @ \
            #    (norm_target_samples - origin_adjust)
            target_simplex_samples = (norm_target_samples - origin_adjust) \
                @ self.transform_matrix.T

            # Draw the transform distances from transform distribution
            transform_dists = self.transform_distrib.sample(num_samples)

            # Add the target to the transform distance to undo distance calc
            pred_simplex_samples = transform_dists + target_simplex_samples

            # Convert the predictor sample back into correct distrib space.
            #self.tf_pred_samples = ((pred_simplex_samples
            #    @ self.transform_matrix) + origin_adjust) * self.total_count
            self.tf_pred_samples = ((pred_simplex_samples
                @ self.transform_matrix) + origin_adjust) * self.total_count

        self.tf_num_samples = num_samples

        # Run once. the saved memory is freed upon this class instance deletion.
        self.tf_sample_sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

    def _create_empirical_predictor_pdf(
        self,
        num_samples=10000,
    ):
        """Creates a probability density function for the predictor output from
        the transform distribution via sampling of the joint distribution and
        using Kernel Density Estimation (KDE).

        Parameters
        ----------
        num_samples : int, optional (default=10000)
            The number of samples to draw from the joint distribution to use to
            get samples of the predictor output to use in fitting the KDE.
        kernel : str
            The kernel identifier to be used by the KDE. Defaults to "tophat".
        """
        target_samples, pred_samples = self.sample(num_samples)
        del target_samples

        # bandwidth is prior standard deviation, but how to select that from
        # data of a distribution of distributions?
        # KDE may be necessary for the continuous case.

        # Could create a DirMult distribution, since we know it will be of that strucutre.
        # TODO uses "UMVUE" for fitting DirMult, must confirm UMVUE is correct!
        self.pred_distrib = self._fit_independent_distrib(
            'DirichletMultinomial',
            pred_samples,
        )

    def _is_prob_distrib(
        self,
        vector,
        rtol=1e-09,
        atol=0.0,
        equal_nan=False,
        axis=1,
    ):
        """Checks if the vector is a valid discrete probability distribution."""
        # check if each row sums to 1
        sums_to_one = np.isclose(vector.sum(axis), 1, rtol, atol, equal_nan)

        # check if all values are w/in range
        in_range = (vector >= 0).all(axis) == (vector <= 1).all(axis)

        return sums_to_one == in_range

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
        origin_adjust = np.zeros(self.transform_matrix.shape[1])
        origin_adjust[0] = 1
        return self.transform_matrix @ (sample - origin_adjust)

    def _transform_from(self, sample):
        """Transforms the sample from probability simplex space into the
        discrete distribtuion space of one dimension more. The orgin adjustment
        is used to move to the correct origin of the discrete distribtuion space.
        """
        # NOTE tensroflow optimization instead here.
        origin_adjust = np.zeros(self.transform_matrix.shape[1])
        origin_adjust[0] = 1
        return (sample @ self.transform_matrix) + origin_adjust

    def sample(self, num_samples, normalize=False):
        """Sample from the estimated joint probability distribution.

        Parameters
        ----------
        normalized_samples : bool, optional (default=False)
            If True, the samples' distributions will be normalized such that
            their values are in the range [0, 1] and all sum to 1, otherwise
            they will be in the range [0, total_count] and sum to total_count.
            This is intended specifically for when the two random variables are
            distributions of distributions (ie. Dirichlet-multinomial).

        Returns
        -------
        (np.ndarray, np.ndarray), shape(samples, input_dim, 2)
            Two arrays of samples from the joint probability distribution of
            the target and the predictor output aligned by samples in their
            first dimension.
        """

        target_samples, pred_samples = self.tf_sample_sess.run(
            [self.tf_target_samples, self.tf_pred_samples],
            feed_dict={self.tf_num_samples: num_samples}
        )

        if self.independent:
            if normalize:
                return (
                    target_samples / self.total_count,
                    pred_samples / self.total_count,
                )
            return target_samples, pred_samples

        # NOTE using Tensorflow while loop to resample and check/rejection
        # would be more optimal. This is next step if necessary.

        # Get any and all indices of non-probability distrib samples
        bad_sample_idx = np.argwhere(np.logical_not(
            self._is_prob_distrib(pred_samples),
        ))
        if len(bad_sample_idx) > 1:
            bad_sample_idx = np.squeeze(bad_sample_idx)
        num_bad_samples = len(bad_sample_idx)
        #adjust_bad = num_bad_samples * (num_samples / (num_samples - num_bad_samples))

        while num_bad_samples > 0:
            print(f'Bad Times: num bad samples = {num_bad_samples}')

            # rerun session w/ enough samples to replace bad samples and some.
            new_pred = self.tf_sample_sess.run(
                self.tf_pred_samples,
                #feed_dict={self.tf_num_samples: np.ceil(num_bad_samples * adjust_bad)},
                feed_dict={self.tf_num_samples: num_bad_samples},
            )

            pred_samples[bad_sample_idx] = new_pred

            bad_sample_idx = np.argwhere(np.logical_not(
                self._is_prob_distrib(pred_samples),
            ))
            if len(bad_sample_idx) > 1:
                bad_sample_idx = np.squeeze(bad_sample_idx)
            num_bad_samples = len(bad_sample_idx)

        if normalize:
            return (
                target_samples / self.total_count,
                pred_samples / self.total_count,
            )
        return target_samples, pred_samples

    def _fit_transform_distrib(self, target, pred, distrib='MultivariateNormal'):
        """Fits and returns the transform distribution."""
        distances = np.array([self._transform_to(pred[i]) - self._transform_to(target[i]) for i in range(len(target))])

        # TODO make some handling of MLE not converging if using adam, ie. to
        # allow the usage of non-convergence mle or not. (should be fine using
        # multivariate gaussian)
        if isinstance(distrib, str):
            if distrib != 'MultivariateNormal':
                raise ValueError(' '.join([
                    'Currently only "MultivariateNormal" is supported for the',
                    'transform distribution as proof of concept.',
                    f'{distrib} is not supported.',
                ]))
            # NOTE logically, mean should be zero given the simplex and
            # distances.  Will need to resample more given

            # deterministically calculate the Maximum Likely multivatiate
            # norrmal TODO need to handle when covariance = 0 and change to an
            # infinitesimal
            return tfp.distributions.MultivariateNormalFullCovariance(
                #np.zeros(pred.shape[1]),
                np.mean(distances, axis=0),
                np.cov(distances, bias=False, rowvar=False),
            )
        if isinstance(distrib, dict):
            if distrib['distrib_id'] != 'MultivariateNormal':
                raise ValueError(' '.join([
                    'Currently only "MultivariateNormal" is supported for the',
                    'transform distribution as proof of concept.',
                    f'{distrib["distrib_id"]} is not a supported value for key',
                    '`distrib_id`.',
                ]))

            self.target_distrib = distribution_tests.mle_adam(
                data=distances,
                **distrib,
            )
        else:
            raise TypeError('`distrib` is expected to be of type `str` or '
                + f'`dict` not `{type(distrib)}`')

    def log_prob(self, values):
        """Log probability density/mass function."""
        # TODO target distrib always easy, just tfp. ... . log_prob()

        # TODO predictor output is not so easy, if only have transform distrib,
        # then able to get empirical pdf of predictor output via samplingG;;
        if self.independent:
            return (
                self.target_distrib.log_prob(values[0]).eval(session=tf.Session()),
                self.transform_distrib.log_prob(values[1]).eval(session=tf.Session()),
            )

        return (
            self.target_distrib.log_prob(values[0]).eval(session=tf.Session()),
            self.pred_distrib.log_prob(values[1]).eval(session=tf.Session()),
        )
