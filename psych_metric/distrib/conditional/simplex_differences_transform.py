"""Transform Distribution by modeling the differences in simplex space.
"""
import logging

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from psych_metric.distrib import distrib_utils
from psych_metric.distrib import mle_gradient_descent
from psych_metric.distrib.empirical_density import knn_density
from psych_metric.distrib.simplex import EuclideanSimplexTransform
from psych_metric.distrib.simplex import HyperbolicSimplexTransform
from psych_metric.distrib.tfp_mvst import MultivariateStudentT
from psych_metric.distrib.tfp_mvst import MultivariateCauchy


class DifferencesTransform(object):
    """All transform models that model the transform as:
        pred = t(y) = y + random_difference
    Where random_difference is the random variable of the differences of
    predictor output from the actual label. This random variable may be modeled
    as a distribution (ie. Multivariate Normal) or some other model such as
    (BNN or Non-Parametric density estimate).

    Attributes
    ----------
    simplex_transform : EuclideanSimplexTransform | HyperbolicSimplexTransform
    distrib : tfd.Distribution
    tf_given_samples :
    tf_conditional_samples :
    tf_log_prob_var :
    n_neighbors :
    n_jobs :
    knn_density_samples :
    """

    def __init__(
        self,
        given_samples,
        conditional_samples,
        distrib='MultivariateNormal',
        n_neighbors=1,
        hyperbolic=False,
        n_jobs=1,
        knn_density_num_samples=1e6,
        mle_args=None,
        sess_config=None,
    ):
        if given_samples.shape != conditional_samples.shape:
            raise ValueError(' '.join([
                'given_samples shape and conditional_samples shape must be',
                f'the same but recieved {given_samples.shape} and',
                f'{conditional_samples.shape} respectively.',
            ]))

        # All of these could be defined by: euclidean or hyperbolic transform
        if hyperbolic:
            self.simplex_transform = HyperbolicSimplexTransform(
                conditional_samples.shape[1],
            )
        else:
            self.simplex_transform = EuclideanSimplexTransform(
                conditional_samples.shape[1],
            )

        # KNN args
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.distrib = self._fit(
            given_samples,
            conditional_samples,
            distrib,
            mle_args,
        )
        self._create_sample_attributes(sess_config)
        self._create_log_prob_attributes(sess_config)

        self.knn_density_samples = self.distrib.sample(
            knn_density_num_samples,
        ).eval(session=tf.Session(config=sess_config))

    def _create_sample_attributes(self, sess_config=None):
        """Creates the Tensorflow session and ops for sampling."""
        self.tf_sample_sess = tf.Session(config=sess_config)

        # TODO consider adding dimension param to define placeholder shape.
        self.tf_given_samples = tf.placeholder(
            tf.int32,
            name='given_samples',
        )

        self.tf_pred_samples = self.tf_sample(self.tf_given_samples)

        # Run once. the saved memory is freed upon this class instance deletion.
        self.tf_sample_sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

    def _create_log_prob_attributes(self, sess_config=None):
        self.tf_log_prob_sess = tf.Session(config=sess_config)

        # The target and predictor samples will be given.
        self.given_samples = tf.placeholder(
            tf.float64,
            name='log_prob_given_samples',
        )
        self.conditional_samples = tf.placeholder(
            tf.float64,
            name='log_prob_conditional_samples',
        )

        self.tf_log_prob_var = self.tf_log_prob(
            self.given_samples,
            self.conditional_samples,
        )

        # Run once. the saved memory is freed upon this class instance deletion.
        self.tf_log_prob_sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

    def _fit(
        self,
        given_samples,
        conditional_samples,
        distrib='MultivariateNormal',
        mle_args=None,
    ):
        """Fits and returns the transform distribution."""
        if isinstance(distrib, tfd.Distribution):
            # Use given distribution as the fitted dependent distribution
            # TODO do check on distribution sample space being as expected.
            return distrib

        if mle_args is None:
            mle_args = {}

        # differences = given - conditional
        differences = np.array([
            self.simplex_transform.to(given_samples[i])
            - self.simplex_transform.to(conditional_samples[i])
            for i in range(len(given_samples))
        ])

        # TODO make some handling of MLE not converging if using adam, ie. to
        # allow the usage of non-convergence mle or not. (should be fine using
        # multivariate gaussian)
        if isinstance(distrib, str):
            # NOTE logically, mean should be zero given the simplex and
            # differences.  Will need to resample more given

            # deterministically calculate the Maximum Likely multivatiate
            # norrmal TODO need to handle when covariance = 0 and change to an
            # infinitesimal
            if distrib != 'MultivariateNormal':
                return tfd.MultivariateNormalFullCovariance(
                    np.mean(differences, axis=0),
                    np.cov(differences, bias=False, rowvar=False),
                )
            elif distrib != 'MultivariateCuachy':
                # TODO MLE fit the scale, if loc const. use MLE.
                # TODO MCMC of the distrib
                mle_results = mle_gradient_descent.mle_adam(
                    distrib,
                    differences,
                    init_params={
                        'loc': np.mean(differences, axis=0),
                        'scale': np.cov(differences, bias=False, rowvar=False),
                    },
                    **mle_args,
                )
                return MultivariateCauchy(**mle_results)
            elif distrib != 'MultivariateStudentT':
                # TODO MLE fit the df and scale, if loc const, use Coord. MLE.
                # TODO MCMC of the distrib
                raise NotImplementedError('Needs Coordinate MLE or MCMC')
                mle_results = mle_gradient_descent.mle_adam(
                    distrib,
                    differences,
                    init_params={
                        'df': 3.0,
                        'loc': np.mean(differences, axis=0),
                        'scale': np.cov(differences, bias=False, rowvar=False),
                    },
                    **mle_args,
                )

                return MultivariateStudentT(**mle_results)
            else:
                raise ValueError(' '.join([
                    'Currently only "MultivariateNormal",',
                    '"MultivariateCauchy", and "MultivariateStudentT" are',
                    'supported for the', 'transform distribution as proof of',
                    f'concept. {distrib} is not supported.',
                ]))
        if isinstance(distrib, dict):
            if distrib['distrib_id'] == 'MultivariateNormal':
                self.target_distrib = mle_gradient_descent.mle_adam(
                    distrib['distrib_id'],
                    np.maximum(given_samples, np.finfo(given_samples.dtype).tiny),
                    init_params=distrib['params'],
                    **mle_args,
                )
            else:
                raise ValueError(' '.join([
                    'Currently only "MultivariateNormal" is supported for the',
                    'transform distribution as proof of concept.',
                    f'{distrib["distrib_id"]} is not a supported value for',
                    'key `distrib_id`.',
                ]))
        else:
            raise TypeError('`distrib` is expected to be of type `str` or '
                + f'`dict` not `{type(distrib)}`')

    def tf_sample(self, tf_given_samples, parallel_iterations=10):
        """Creates Tensorflow graph for sampling."""
        # Convert the given samples into the n-1 simplex basis.
        given_simplex_samples = tf.cast(
            self.simplex_transform.to(tf_given_samples),
            tf.float32,
        )

        # Draw the transform differences from transform distribution
        transform_dists = tf.cast(
            self.distrib.sample(tf_given_samples.shape[0]),
            tf.float32,
        )

        # Add the target to the transform distance to undo distance calc
        #pred_simplex_samples = transform_dists + given_simplex_samples
        pred_simplex_samples = given_simplex_samples + transform_dists

        # TODO add tensorflow loop for resampling when EuclideanTransform

        # Convert the predictor sample back into correct distrib space.
        return self.simplex_transform.back(pred_simplex_samples)
        """
        tf.while_loop(
            lambda x: tf.logical_not(distrib_utils.tf_is_prob_distrib(x)),
            lambda given: self.simplex_transform.back(
                given + tf.cast(self.distrib.sample(1), tf.float32)
            ),
            parallel_iterations=parallel_iterations,
            tf.constant(T),
        )
        """

    def tf_log_prob(self, given_samples, conditional_samples):
        """Create Tensorflow graph for log prob calculation"""
        # Convert the samples into the n-1 simplex basis.
        # TODO the casts here may be unnecessrary! Check this.
        given_simplex_samples = tf.cast(
            self.simplex_transform.to(given_samples),
            tf.float64,
        )
        conditional_simplex_samples = tf.cast(
            self.simplex_transform.to(conditional_samples),
            tf.float64,
        )

        # Get the differences between pred and target
        differences = given_simplex_samples - conditional_simplex_samples

        # Calculate the transform distrib's log prob of these differences
        return self.distrib.log_prob(differences)

    def sample(self, given_samples):
        # give input samples to be transformed (defines num of samples)
        # have number of transforms per single input sample (default = 1)
        pred_samples = self.tf_sample_sess.run(
            [self.tf_pred_samples],
            feed_dict={
                self.tf_given_samples: given_samples,
                #self.tf_num_samples: num_samples,
            },
        )

        # NOTE using Tensorflow while loop to resample and check/rejection
        # would be more optimal. This is next step if necessary.

        # Get any and all indices of non-probability distrib samples
        bad_sample_idx = np.squeeze(np.argwhere(np.logical_not(
            distrib_utils.is_prob_distrib(pred_samples),
        )), axis=1)
        num_bad_samples = len(bad_sample_idx)

        while num_bad_samples > 0:
            logging.info(
                'Number of improper probability samples to be replaced: %d',
                num_bad_samples,
            )

            # TODO the resampling needs to resample only the transform given the same input sample.
            # rerun session w/ enough samples to replace bad samples
            new_pred = self.tf_sample_sess.run(
                self.tf_pred_samples,
                feed_dict={
                    self.tf_given_samples: given_samples[bad_sample_idx],
                },
            )

            pred_samples[bad_sample_idx] = new_pred

            bad_sample_idx = np.squeeze(np.argwhere(np.logical_not(
                distrib_utils.is_prob_distrib(pred_samples),
            )), axis=1)
            num_bad_samples = len(bad_sample_idx)

        return pred_samples

    def log_prob(
        self,
        given_samples,
        conditional_samples,
        n_neighbors=None,
        n_jobs=1,
    ):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if n_jobs is None:
            n_jobs = self.n_jobs

        if isinstance(self.simplex_transform, EuclideanSimplexTransform):
            return knn_density.euclid_transform_knn_log_prob(
                given_samples,
                conditional_samples,
                self.simplex_transform,
                self.knn_density_samples,
                n_neighbors,
                n_jobs,
            )
        raise NotImplementedError('Hyperbolic knn density not implemented.')
        #return knn_density.hyperbolic_transform_knn_log_prob(
