"""Bayesian Joint Probability of target distribution and predictor output
distribution. Assumes that the predictor output is dependent on the target
distribution and that the prediction output can be obtained with some error by
a transformation function of the target distribution.
"""
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import scipy
from sklearn.neighbors import BallTree
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric import distribution_tests
from psych_metric import mvst
from psych_metric.distrib import mle_gradient_descent

# TODO handle continuous distrib of continuous distrib, only discrete atm.


def transform_to(sample, transform_matrix, origin_adjust=None):
    """Transforms the sample from discrete distribtuion space into the
    probability simplex space of one dimension less. The orgin adjustment
    is used to move to the correct origin of the probability simplex space.
    """
    if origin_adjust is None:
        origin_adjust = np.zeros(transform_matrix.shape[1])
        origin_adjust[0] = 1
    return transform_matrix @ (sample - origin_adjust)


def transform_from(sample, transform_matrix, origin_adjust=None):
    if origin_adjust is None:
        origin_adjust = np.zeros(transform_matrix.shape[1])
        origin_adjust[0] = 1
    return (sample @ transform_matrix) + origin_adjust


def knn_log_prob(pred, num_classes, knn_tree, k, knn_pdf_num_samples=int(1e6)):
    """Empirically estimates the predictor log probability using K Nearest
    Neighbords.
    """
    if len(pred.shape) == 1:
        # single sample
        pred = pred.reshape(1, -1)

    #radius = self.knn_tree.query(pred, k)[0][:, -1]
    radius = knn_tree.query(pred, k)[0][:, -1]

    # log(k) - log(n) - log(volume)
    log_prob = np.log(k) - np.log(knn_pdf_num_samples)

    # calculate the n-1 sphere volume being contained w/in the n-1 simplex
    n = num_classes - 1
    log_prob -= n * (np.log(np.pi) / 2 + np.log(radius)) \
        - scipy.special.gammaln(n / 2 + 1)

    return log_prob


def transform_knn_log_prob_single(
    trgt,
    pred,
    transform_knn_dists,
    k,
    origin_adjust,
    transform_matrix,
):
    # Find valid differences from saved set: `self.transform_knn_dists`
    # to test validity, convert target sample to simplex space.
    simplex_trgt = transform_to(trgt, transform_matrix, origin_adjust)

    # add differences to target & convert back to full dimensional space.
    dist_check = transform_from(
        transform_knn_dists + simplex_trgt,
        transform_matrix,
        origin_adjust,
    )

    # Check which are valid samples. Save indices or new array
    valid_dists = transform_knn_dists[
        np.where(distirbution_tests.is_prob_distrib(dist_check))[0]
    ]

    # Fit BallTree to the differences valid to the specific target.
    knn_tree = BallTree(valid_dists)

    # Get distance between actual sample pair of target and pred
    actual_dist = transform_to(pred, transform_matrix, origin_adjust) - simplex_trgt

    # Estimate the log probability.
    return knn_log_prob(actual_dist, transform_matrix.shape[1], knn_tree, k)


class SupervisedJointDistrib(object):
    """Bayesian distribution fitting of a joint probability distribution whose
    values correspond to the supervised learning context with target data and
    predictor output data.

    Attributes
    ----------
    independent : bool
        True if the random variables are indpendent of one anothers, False
        otherwise. Default is False
    transform_matrix : np.ndarray
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
    knn_tree : sklearn.neighbors.BallTree
        The BallTree that is used to calculate the empirical density of the
        predictor probability density function when the predictor random
        variable is dependent on the target.
    knn_pdf_num_samples : int
        Number of samples used to estimate the predictor pdf when predictor is
        dependent on target.
    sample_dim : int, optional
        The number of dimensions of a single sample of both the target and
        predictor distribtutions.
    """

    def __init__(
        self,
        target_distrib,
        transform_distrib,
        target=None,
        pred=None,
        data_type='nominal',
        independent=False,
        num_neighbors=10,
        sample_dim=None,
        total_count=None,
        tf_sess_config=None,
        mle_args=None,
        knn_num_samples=int(1e6),
        dtype=np.float32,
        processes=16,
    ):
        """
        Parameters
        ----------
        target_distrib : str | dict | tfp.distribution.Distribution
            Either the parameters to a  distribution to be used as the fitted
            distribution the target data, or the actual distribution to be
            used.
        transform_distrib : str | dict | tfp.distribution.Distribution
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
        mle_args : dict, optional
            Dictionary of arguments for the maximum likelihood estimation
            method used. Contains an `optimizer_id` key with a str identifier
            of which optimization method is used: 'adam', 'nadam',
            'nelder-mead', 'simulated_annealing', 'nuts', 'random_walk'
        knn_num_samples : int
            number of samples to draw for the KNN density estimate. Defaults to
            int(1e6).
        dtype : type, optional
            The data type of the data. If a single type, all data is assumed to
            be of that type. Defaults to np.float32
        num_params : int
            The total number of parameters of the entire joint probability
            distribution model.
        num_params_target : int
            The number of parameters of the target distribution
        num_params_transform : int
            The number of parameters of the transform distribution
        """
        self.independent = independent
        self.num_neighbors = num_neighbors
        self.dtype = dtype
        self.processes = processes

        if not isinstance(total_count, int):
            # Check if any distribs are DirMult()s. They need total_counts
            if (
                isinstance(target_distrib, tfp.distributions.DirichletMultinomial)
                or (
                    isinstance(target_distrib, dict)
                    and 'DirichletMultinomial' == target_distrib['distrib_id']
                )
                or isinstance(transform_distrib, tfp.distributions.DirichletMultinomial)
                or (
                    isinstance(transform_distrib, dict)
                    and 'DirichletMultinomial' == transform_distrib['distrib_id']
                )
            ):
                raise TypeError(' '.join([
                    '`total_count` of type `int` must be passed in order to',
                    'use DirichletMultinomial distribution as the target',
                    f'distribution, but recieved type {type(total_count)}',
                    'instead.',
                ]))
        self.total_count = total_count

        # Set the class' sample_dim and transform_matrix
        if target is not None and pred is not None:
            if target.shape != pred.shape:
                raise ValueError(' '.join([
                    '`target.shape` and `pred.shape` must be the same shape.',
                    f'Instead recieved shapes {target.shape} and {pred.shape}.',
                ]))

            self.sample_dim = target.shape[1]

            # Get transform matrix from data
            self.transform_matrix = self._get_change_of_basis_matrix(
                target.shape[1]
            )

            # Create the origin adjustment for going to and from n-1 simplex.
            self.origin_adjust = np.zeros(target.shape[1])
            self.origin_adjust[0] = 1
        elif isinstance(sample_dim, int):
            self.sample_dim = sample_dim
            self.transform_matrix = self._get_change_of_basis_matrix(sample_dim)

            # Create the origin adjustment for going to and from n-1 simplex.
            self.origin_adjust = np.zeros(sample_dim)
            self.origin_adjust[0] = 1
        else:
            TypeError(' '.join([
                '`target` and `pred` must be provided together, otherwise',
                '`sample_dim` must be given instead, along with',
                '`target_distrib` and `transform_distrib` given explicitly as',
                'an already defined distribution each.',
            ]))

        # NOTE given the nature of the distribs being separate, the fitting of
        # them within this class is a convenience for the research.
        # Practically this class would only store the target and the
        # conditional prob related code (simplex conversion and estimate of the
        # transform function)

        # Fit the data (simply modularizes the fitting code,)
        self.fit(
            target_distrib,
            transform_distrib,
            target,
            pred,
            independent,
            mle_args,
            knn_num_samples,
        )

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

    def _fit_independent_distrib(
        self,
        distrib,
        data=None,
        mle_args=None,
    ):
        """Fits the given data with an independent distribution."""
        if isinstance(distrib, tfp.distributions.Distribution):
            # Use given distribution as the fitted distribution.
            return distrib
        elif isinstance(distrib, str):
            distrib = distrib.lower()

            # By default, use UMVUE of params of given data. No MLE fitting.
            if data is None:
                raise ValueError('`data` must be provided when `distrib` '
                    + 'is of type `str`')

            if distrib == 'dirichletmultinomial':
                total_count = np.sum(data, axis=1).max(0)

                if mle_args:
                    data = data.astype(np.float32)
                    mle_results = mle_gradient_descent(
                        distrib,
                        np.maximum(data, np.finfo(data.dtype).tiny),
                        init_params={
                            'total_count': total_count,
                            'concentration': np.mean(data, axis=0),
                        },
                        const_params=['total_count'],
                        **mle_args,
                    )

                    return  tfp.distributions.DirichletMultinomial(
                        **mle_results[0].params
                    )

                return  tfp.distributions.DirichletMultinomial(
                    total_count,
                    np.mean(data, axis=0), #/ total_count,
                )
            elif distrib == 'dirichlet':
                if mle_args:
                    # TODO need to 1) be given a dtype, 2) enforce that in all
                    # data and tensors.
                    data = data.astype(np.float32)
                    mle_results = mle_gradient_descent(
                        distrib,
                        np.maximum(data, np.finfo(data.dtype).tiny),
                        init_params={'concentration': np.mean(data, axis=0)},
                        **mle_args,
                    )

                    if mle_args['alt_distrib']:
                        return  tfp.distributions.Dirichlet(
                            mle_results[0].params['mean'] * mle_results[0].params['precision']
                        )

                    return  tfp.distributions.Dirichlet(
                        **mle_results[0].params
                    )

                return  tfp.distributions.Dirichlet(np.mean(data, axis=0))
            elif distrib == 'multivariatenormal':
                return tfp.distributions.MultivariateNormalFullCovariance(
                    np.mean(data, axis=0),
                    np.cov(data, bias=False, rowvar=False),
                )
            elif distrib == 'multivariatestudentt':
                raise NotImplementedError('Need to return a non tfp distrib')
                # TODO ensure all is well in mle_adam and return the tfp
                # MultivariateStudentT Linear Operator.

                # initial df is 3 for loc = mean and Covariance exist.
                # initial random scale is cov * (df -2) / df / 2
                # as a poor attempt to ge a matrix in the proper range of
                # values of scale.

                if mle_args:
                    mle_results = mle_gradient_descent(
                        distrib,
                        data,
                        init_params={
                            'df': 3.0,
                            'loc': np.mean(data, axis=0),
                            'covariance_matrix': np.cov(data, bias=False, rowvar=False),
                        },
                        **mle_args,
                    )

                    # TODO handle returning a NON tfp distrib, as MVSTLinOp
                    # relies on scale, rather than Sigma.
                    #return  tfp.distributions.MultivariateStudentTLinearOperator(
                    #    **mle_results[0].params
                    #)
                    raise NotImplementedError('Need to return a non tfp distrib')

                    return  mvst.MultivariateStudentT(**mle_results[0].params)
            else:
                raise ValueError(' '.join([
                    'Currently only "Dirichlet", "DirichletMultinomial",',
                    '"MultivariateNormal", or "MultivariateStudentT" for,',
                    'independent distributions are supported as proof of',
                    'concept.',
                ]))
        elif isinstance(distrib, dict):
            # If given a dict, use as initial parameters and fit with MLE
            data = data.astype(np.float32)
            if (
                distrib['distrib_id'] == 'DirichletMultinomial'
                or distrib['distrib_id'] == 'Dirichlet'
                or distrib['distrib_id'] == 'MultivariateNormal'
            ):
                return  mle_gradient_descent(
                    distrib['distrib_id'],
                    np.maximum(data, np.finfo(data.dtype).tiny),
                    init_params=distrib['params'],
                    **mle_args,
                )
            else:
                raise ValueError(' '.join([
                    'Currently only "DirichletMultinomial", "Dirichlet", or',
                    '"MultivariateNormal" for ' 'independent distributions',
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
            if isinstance(
                self.target_distrib,
                tfp.distributions.DirichletMultinomial,
            ):
                norm_target_samples = self.tf_target_samples / self.total_count
            else:
                norm_target_samples = self.tf_target_samples

            # Convert the normalized target samples into the n-1 simplex basis.
            target_simplex_samples = tf.cast(
                (norm_target_samples - self.origin_adjust) @ self.transform_matrix.T,
                tf.float32,
            )

            # Draw the transform differences from transform distribution
            transform_dists = tf.cast(
                self.transform_distrib.sample(num_samples),
                tf.float32,
            )

            # Add the target to the transform distance to undo distance calc
            pred_simplex_samples = transform_dists + target_simplex_samples

            # Convert the predictor sample back into correct distrib space.
            self.tf_pred_samples = ((pred_simplex_samples
                @ self.transform_matrix) + self.origin_adjust)

            if isinstance(
                self.target_distrib,
                tfp.distributions.DirichletMultinomial,
            ):
                self.tf_pred_samples = self.tf_pred_samples * self.total_count

        self.tf_num_samples = num_samples

        # Run once. the saved memory is freed upon this class instance deletion.
        self.tf_sample_sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

    def _create_log_joint_prob_attributes(
        self,
        independent,
        sess_config=None,
    ):
        if independent:
            self.tf_log_prob_sess = None
            self.joint_log_prob = None
            self.log_prob_target_samples = None
            self.log_prob_pred_samples = None
            return

        self.tf_log_prob_sess = tf.Session(config=sess_config)

        # The target and predictor samples will be given.
        log_prob_target_samples = tf.placeholder(
            tf.float64,
            name='log_prob_target_samples',
        )
        log_prob_pred_samples = tf.placeholder(
            tf.float64,
            name='log_prob_pred_samples',
        )

        # Normalize the samples for use with dependent transform
        if isinstance(
            self.target_distrib,
            tfp.distributions.DirichletMultinomial,
        ):
            norm_target_samples = log_prob_target_samples / self.total_count
        else:
            norm_target_samples = log_prob_target_samples

        if isinstance(
            self.transform_distrib,
            tfp.distributions.DirichletMultinomial,
        ):
            norm_pred_samples = log_prob_pred_samples / self.total_count
        else:
            norm_pred_samples = log_prob_pred_samples


        # Convert the normalized samples into the n-1 simplex basis.
        # TODO the casts here may be unnecessrary! Check this.
        target_simplex_samples = tf.cast(
            (norm_target_samples - self.origin_adjust) @ self.transform_matrix.T,
            tf.float64,
        )
        pred_simplex_samples = tf.cast(
            (norm_pred_samples - self.origin_adjust) @ self.transform_matrix.T,
            tf.float64,
        )

        # Get the differences between pred and target
        differences = pred_simplex_samples - target_simplex_samples

        # Calculate the transform distrib's log prob of these differences
        self.joint_log_prob = self.transform_distrib.log_prob(differences)

        # Save the placeholders as class attributes
        self.log_prob_target_samples = log_prob_target_samples
        self.log_prob_pred_samples = log_prob_pred_samples

        # Run once. the saved memory is freed upon this class instance deletion.
        self.tf_log_prob_sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

        return

    def _create_empirical_predictor_pdf(
        self,
        num_samples=int(1e6),
        independent=False,
    ):
        """Creates a probability density function for the predictor output from
        the transform distribution via sampling of the joint distribution and
        using Kernel Density Estimation (KDE).

        Parameters
        ----------
        num_samples : int, optional (default=int(1e6))
            The number of samples to draw from the joint distribution to use to
            get samples of the predictor output to use in fitting the KDE.
        kernel : str
            The kernel identifier to be used by the KDE. Defaults to "tophat".
        """
        if independent:
            self.knn_tree = None
            self.knn_pdf_num_samples = None
        else:
            target_samples, pred_samples = self.sample(num_samples)
            del target_samples

            self.knn_tree = BallTree(pred_samples)
            self.knn_pdf_num_samples = num_samples

    def _create_knn_transform_pdf(
        self,
        independent=False,
        num_samples=int(1e6),
    ):
        """KNN density estimate for the transform distribution whose space is
        that of the differences of valid points within the probability simplex.

        The Tensorflow Probability log prob for the transform will extremely
        under estimate the log probability when the gaussian (or any distrib
        for the transform) has a significant amount of its density outside of
        the bounds of the simplex. As such it is necessary to calculate the log
        prob another way, hence this KNN density estimate approach.
        """
        if independent:
            # TODO Set the necessary attributes to None, so they exist but are empty.
            self.knn_pdf_num_samples = None
            self.transform_knn_dists = None
        else:
            self.knn_pdf_num_samples = num_samples
            self.transform_knn_dists = self.transform_distrib.sample(num_samples).eval(session=tf.Session())

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
        return self.transform_matrix @ (sample - self.origin_adjust)

    def _transform_from(self, sample):
        """Transforms the sample from probability simplex space into the
        discrete distribtuion space of one dimension more. The orgin adjustment
        is used to move to the correct origin of the discrete distribtuion space.
        """
        # NOTE tensroflow optimization instead here.
        return (sample @ self.transform_matrix) + self.origin_adjust

    def _fit_transform_distrib(
        self,
        target,
        pred,
        distrib='MultivariateNormal',
        mle_args=None,
        zero_loc=False,
    ):
        """Fits and returns the transform distribution."""
        differences = np.array([self._transform_to(pred[i]) - self._transform_to(target[i]) for i in range(len(target))])

        # TODO make some handling of MLE not converging if using adam, ie. to
        # allow the usage of non-convergence mle or not. (should be fine using
        # multivariate gaussian)
        if isinstance(distrib, str):
            if (
                distrib != 'MultivariateNormal'
                or distrib != 'MultivariateCauchy'
                or distrib != 'MultivariateStudentT'
            ):
                raise ValueError(' '.join([
                    'Currently only "MultivariateNormal",',
                    '"MultivariateCauchy", and "MultivariateStudentT" are',
                    'supported for the', 'transform distribution as proof of',
                    f'concept. {distrib} is not supported.',
                ]))
            # NOTE logically, mean should be zero given the simplex and
            # differences.  Will need to resample more given

            # deterministically calculate the Maximum Likely multivatiate
            # norrmal TODO need to handle when covariance = 0 and change to an
            # infinitesimal
            return tfp.distributions.MultivariateNormalFullCovariance(
                #np.zeros(pred.shape[1]),
                np.mean(differences, axis=0),
                np.cov(differences, bias=False, rowvar=False),
            )
        if isinstance(distrib, dict):
            if distrib['distrib_id'] != 'MultivariateNormal':
                raise ValueError(' '.join([
                    'Currently only "MultivariateNormal" is supported for the',
                    'transform distribution as proof of concept.',
                    f'{distrib["distrib_id"]} is not a supported value for key',
                    '`distrib_id`.',
                ]))

            self.target_distrib = mle_gradient_descent(
                distrib['distrib_id'],
                np.maximum(target, np.finfo(target.dtype).tiny),
                init_params=distrib['params'],
                **mle_args,
            )
        else:
            raise TypeError('`distrib` is expected to be of type `str` or '
                + f'`dict` not `{type(distrib)}`')

    def _knn_log_prob(self, pred, knn_tree, k=None):
        if k is None:
            k = self.num_neighbors

        return knn_log_prob(
            pred,
            self.transform_matrix.shape[1],
            knn_tree,
            k,
            self.knn_pdf_num_samples,
        )

    def _transform_knn_log_prob(self, target, pred, k=None):
        if k is None:
            k = self.num_neighbors
        if processes is None:
            processes = self.processes

        with Pool(processes=processes) as pool:
            log_prob = pool.starmap(
                transform_knn_log_prob_single,
                zip(
                    target,
                    pred,
                    [self.transform_knn_dists] * len(target),
                    [k] * len(target),
                    [self.origin_adjust] * len(target),
                    [self.transform_matrix] * len(target),
                ),
            )

        return np.array(log_prob)

    @property
    def num_params(self):
        return self.target_num_params + self.transform_num_params

    def fit(
        self,
        target_distrib,
        transform_distrib,
        target,
        pred,
        independent=False,
        mle_args=None,
        knn_num_samples=int(1e6),
        tf_sess_config=None
    ):
        """Fits the target and transform distributions to the data."""
        # TODO check if the target and pred match the distributions' sample
        # space

        # TODO parallelize the fitting of the target and predictor distribs

        # Fit the target data
        self.target_distrib = self._fit_independent_distrib(
            target_distrib,
            target,
            mle_args,
        )

        # Set the number of parameters for the target distribution
        self.target_num_params = distribution_tests.get_num_params(
            target_distrib,
            self.sample_dim,
        )

        # Fit the transformation function of the target to the predictor output
        if self.independent:
            self.transform_distrib = self._fit_independent_distrib(
                transform_distrib,
                pred,
                mle_args,
            )
        elif isinstance(transform_distrib, tfp.distributions.Distribution):
            # Use given distribution as the fitted dependent distribution
            self.transform_distrib = transform_distrib
        else:
            self.transform_distrib = self._fit_transform_distrib(
                target,
                pred,
                transform_distrib,
            )

        # Set the number of parameters for the transform distribution
        self.transform_num_params = distribution_tests.get_num_params(
            transform_distrib,
            self.sample_dim,
        )

        # Create the Tensorflow session and ops for sampling
        self._create_sampling_attributes(tf_sess_config)
        self._create_log_joint_prob_attributes(
            self.independent,
            tf_sess_config,
        )
        #self._create_empirical_predictor_pdf(independent=self.independent)

        # create the transform log_prob knn
        self._create_knn_transform_pdf(self.independent, knn_num_samples)


    def sample(self, num_samples, normalize=False):
        """Sample from the estimated joint probability distribution.

        Parameters
        ----------
        normalize : bool, optional (default=False)
            If True, the samples' distributions will be normalized such that
            their values are in the range [0, 1] and all sum to 1, otherwise
            they will be in the range [0, total_count] and sum to total_count.
            This is intended specifically for when the two random variables are
            distributions of distributions (ie. Dirichlet-multinomial).

            Ensure the elements sum to one (under assumption they sum to total
            counts.)

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
                if isinstance(self.target_distrib, tfp.distributions.DirichletMultinomial):
                    target_samples = target_samples / self.total_count

                if isinstance(self.transform_distrib, tfp.distributions.DirichletMultinomial):
                    pred_samples = pred_samples / self.total_count

            return target_samples, pred_samples

        # NOTE using Tensorflow while loop to resample and check/rejection
        # would be more optimal. This is next step if necessary.

        # Get any and all indices of non-probability distrib samples
        bad_sample_idx = np.argwhere(np.logical_not(
            distribution_tests.is_prob_distrib(pred_samples),
        ))
        if len(bad_sample_idx) > 1:
            bad_sample_idx = np.squeeze(bad_sample_idx)
        num_bad_samples = len(bad_sample_idx)

        while num_bad_samples > 0:
            print(f'Bad Times: num bad samples = {num_bad_samples}')

            # rerun session w/ enough samples to replace bad samples and some.
            new_pred = self.tf_sample_sess.run(
                self.tf_pred_samples,
                feed_dict={self.tf_num_samples: num_bad_samples},
            )

            pred_samples[bad_sample_idx] = new_pred

            bad_sample_idx = np.argwhere(np.logical_not(
                distribution_tests.is_prob_distrib(pred_samples),
            ))
            if len(bad_sample_idx) > 1:
                bad_sample_idx = np.squeeze(bad_sample_idx)
            num_bad_samples = len(bad_sample_idx)

        if normalize:
            if isinstance(self.target_distrib, tfp.distributions.DirichletMultinomial):
                target_samples = target_samples / self.total_count

                # NOTE assumes that predictors samples is of the same
                # output as target, when target is a DirichletMultinomial.
                pred_samples = pred_samples / self.total_count

        return target_samples, pred_samples

    def log_prob(
        self,
        target,
        pred,
        num_neighbors=None,
        return_individuals=False,
    ):
        """Log probability density/mass function calculation for the either the
        joint probability by default or the individual random variables.

        Parameters
        ----------
        target : np.ndarray
            The samples who serve as the independent random variable.
        pred : np.ndarray
            The samples who serve as the 2nd random variable, either dependent
            or independent.
        num_neighbors : int, optional
            The number of neighbors to use in the K Nearest Neighbors density
            estimate of the probability of the 2nd random variable's pdf. Only
            used if the 2nd random variable is dependent upon 1st, meaning we
            use a stochastic transform funciton.
        joint : bool
            If True then returns the joint log probability p(target, pred),
            otherwise returns a tuple/list of the two random variables
            individual log probabilities.

        Returns
        -------
            Returns the joint log probability as a float when `joint` is True
            or returns the individual random variables unconditional log
            probability when `joint` is False.
        """
        if (
            isinstance(self.target_distrib, tfp.distributions.DirichletMultinomial)
            or isinstance(self.target_distrib, tfp.distributions.Dirichlet)
        ):
            target = target.astype(self.dtype)
            target = np.maximum(target, np.finfo(target.dtype).tiny)

        if self.independent:
            # If independent, then just measure the MLE of the distribs separately.
            if (
                isinstance(self.transform_distrib, tfp.distributions.DirichletMultinomial)
                or isinstance(self.transform_distrib, tfp.distributions.Dirichlet)
            ):
                pred = pred.astype(self.dtype)
                pred = np.maximum(pred, np.finfo(pred.dtype).tiny)

            with tf.Session() as sess:
                log_prob_pair = sess.run((
                    self.target_distrib.log_prob(target),
                    self.transform_distrib.log_prob(pred)
                ))

            if return_individuals:
                return (
                    log_prob_pair[0] + log_prob_pair[1],
                    log_prob_pair[0],
                    log_prob_pair[1],
                )

            return log_prob_pair[0] + log_prob_pair[1]

        log_prob_target = self.target_distrib.log_prob(target).eval(
            session=tf.Session(),
        )

        # Calculate the log prob of the stochastic transform function
        #log_prob_pred = self.tf_log_prob_sess.run(
        #    self.joint_log_prob,
        #    feed_dict={
        #        self.log_prob_target_samples: target,
        #        self.log_prob_pred_samples: pred,
        #    },
        #)

        log_prob_pred = self._transform_knn_log_prob(target, pred)

        if return_individuals:
            return (
                #log_prob_target + log_prob_pred,
                None,
                log_prob_target,
                log_prob_pred,
            )

        return log_prob_target + log_prob_pred

    def info_criterion(self, mle, criterions='bic', num_samples=None, data=None):
        """Calculate information criterions.

        Parameters
        ----------
        mle : float | np.ndarray
            The pre-calculated maximum likelihood estimate.
        criterions : str | array-like, optional
            array-like of strings that indicate which criterions to be
            calculated from the following: 'bic', 'aic', 'hqc'. Defaults to
            'bic' for Bayesian Information Criterion.
        num_samples : int, optional
            Only necessary when criterions is or includes `bic` or 'hqc' to
            calculate the Hannan-Quinn information criterion. This is
            unnecessary if data is provided.
        data : tuple, optional
            Tuple of two np.ndarrays containing the data pertaining to the
            target and the predictor respectively. They must be of the shape
            matching their respective distribution's sample space. The usage of
            this parameter overrides the usage of MLE and the mle will be
            calculated from the log probability of the data.

        Returns
        -------
            A dict of floats for each criterion calculated.
        """
        # TODO calculate log prob if data is given

        # TODO need to add num_params property (target num params, transform
        # num params) and num params of sjd = sum of that.
        info_criterion = {}

        # TODO type checking and exception raising

        if 'bic' in criterions:
            info_criterion['bic'] = distribution_tests.bic(
                mle,
                self.num_params,
                num_samples,
            )

        if 'aic' in criterions:
            info_criterion['aic'] = distribution_tests.aic(
                mle,
                self.num_params,
            )

        if 'hqc' in criterions:
            info_criterion['hqc'] = distribution_tests.hqc(
                mle,
                self.num_params,
                num_samples,
            )

        return info_criterion

    def entropy(self):
        # TODO calculate / approximate the entropy of the joint distribution
        raise NotImplementedError
