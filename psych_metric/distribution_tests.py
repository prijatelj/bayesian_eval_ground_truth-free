"""Functions for performing distribution model selection."""
import logging
import math
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def aic(mle, num_params):
    """Akaike information criterion."""
    return 2 * (num_params - math.log(mle))


def bic(mle, num_params, num_samples):
    """Bayesian information criterion. Approx. Bayes Factor"""
    return num_params * math.log(num_samples) - 2 * math.log(mle)


def hqc(mle, num_params, num_samples):
    """Hannan-Quinn information criterion."""
    return 2 * (num_params * math.log(math.log(num_samples)) - mle)


def dic(likelihood_function, num_params, num_samples, mle=None, mean_lf=None):
    """Deviance information criterion.

    Parameters
    ----------
    mle : float
        the Maximum Likelihood Estimate to repreesnt the distribution's
        likelihood function.
    """
    raise NotImplementedError('Need to implement finding the MLE from '
        + 'Likelihood function, and the expected value of the likelihood '
        + 'function.')

    if mean_lf is None:
        raise NotImplementedError('Need to implement finding the expected '
            + 'value from Likelihood function.')
    if mle is None:
        raise NotImplementedError('Need to implement finding the MLE from '
            + 'Likelihood function.')

    # DIC = 2 * (pd - D(theta)), where pd is spiegelhalters: pd= E(D(theta)) - D(E(theta))
    # return 2 * (expected_value_likelihood_func - math.log(mle) - math.log(likelihood_funciton))


# NOTE MLE search over params  could be done in SHADHO instead
def mle_adam(
    distrib_id,
    data,
    init_params=None,
    optimizer_args=None,
    num_top_likelihoods=1,
    max_iter=10000,
    tol_param=1e-8,
    tol_loss=1e-8,
    tol_grad=1e-8,
    tb_summary_dir=None,
    random_seed=None,
    name='MLE_adam',
    sess_config=None,
):
    """Uses tensorflow's ADAM optimizer to search the parameter space for MLE.

    Parameters
    ----------
    distrib_id : str
        Name of the distribution whoe MLE is being found.
    data : np.ndarray
    init_params : dict, optional
        The initial parameters of the distribution. Otherwise, selected randomly.
    optimizer_args : dict, optional
    num_top_likelihoods : int, optional
        The number of top best likelihoods and their respective parameters.
    tb_summary_dir : str, optional
        directory path to save TensorBoard summaries.
    name : str
        Name prefixed to Ops created by this class.

    Returns
    -------
    top_likelihoods
        the `num_top_likelihoods` likelihoods and their respective parameters
        in decending order.
    """
    if optimizer_args is None:
        optimizer_args = {}
    if random_seed:
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

    # tensor to hold the data
    with tf.name_scope(name) as scope:
        if isinstance(data, np.ndarray):
            data = tf.placeholder(dtype=tf.float32, name='data')

        # create distribution and dict of the distribution's parameters
        distrib, params = get_distrib_param_vars(distrib_id, init_params)

        # TODO why negative? is this necessary? should it be a user passed flag?
        # because ths is the minimized loss, and we want the Maximumg Likelihood
        neg_log_likelihood = -1.0 * tf.reduce_sum(
            distrib.log_prob(value=data),
            name='log_likelihood_sum',
        )

        # Create optimizer
        optimizer = tf.train.AdamOptimizer(**optimizer_args)
        global_step = tf.Variable(0, name='global_step', trainable=False)

        grad = optimizer.compute_gradients(
            neg_log_likelihood,
            list(params.values()),
        )
        train_op = optimizer.apply_gradients(grad, global_step)

        if tb_summary_dir:
            # Visualize the gradients
            for g in grad:
                tf.summary.histogram(f'{g[1].name}-grad', g[0])

            # TODO Visualize the values of params

            summary_op = tf.summary.merge_all()

    with tf.Session(config=sess_config) as sess:
        # Build summary operation
        if tb_summary_dir:
            summary_writer = tf.summary.FileWriter(
                os.path.join(
                    tb_summary_dir,
                    str(datetime.now()).replace(':', '-').replace(' ', '_'),
                ),
                sess.graph
            )

        sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

        # MLE loop
        top_likelihoods = []

        params_history = []
        loss_history = []
        grad_history = []

        i = 1
        while True:
            # get likelihood and params
            iter_results = sess.run({
                'train_op': train_op,
                'neg_log_likelihood': neg_log_likelihood,
                'params': params,
                'summary_op': summary_op,
            })

            if iter_results['neg_log_likelihood'] < top_likelihoods[-1]:
                # update top likelihoods and their respective params
                if num_top_likelihoods <= 1:
                    top_likelihoods[0] = (
                        iter_results['neg_log_likelihood'],
                        iter_results['params'],
                    )
                else:
                    top_likelihoods.append((
                        iter_results['neg_log_likelihood'],
                        iter_results['params'],
                    ))
                    top_likelihoods = sorted(top_likelihoods)

                    if len(top_likelihoods) > num_top_likelihoods:
                        del top_likelihoods[-1]

            # Save observed vars of interest
            params_history.append(iter_results['params'])
            loss_history.append(iter_results['neg_log_likelihood'])
            grad_history.append(iter_results['grad'])

            if tb_summary_dir:
                # Write summary update
                summary_writer.add_summary(iter_results['summary_op'], i)
                summary_writer.flush()

            # Calculate Termination Conditions
            # TODO Does the values need sorted by keys first?
            param_diff = np.subtract(
                iter_results['params'].values,
                params_history[-1].values,
            )
            if params_history and np.linalg.norm(param_diff) < tol_param:
                logging.info('Parameter convergence in %d iterations.', i)
                break

            if loss_history and np.abs(iter_results['loss'] - loss_history[-1]) < tol_param:
                logging.info('Loss convergence in %d iterations.', i)
                break

            if grad_history and np.linalg.norm(iter_results['grad']) < tol_param:
                logging.info('Gradient convergence in %d iterations.', i)
                break

            if i >= max_iter:
                logging.info('Maimum iterations (%d) reached without convergence.', max_iter)
                break

            i += 1

    if tb_summary_dir:
        summary_writer.close()

    return top_likelihoods
    # return top_likelihoods, {'params': params_history, 'loss': loss_history, 'grad': grad_history}


def get_distrib_param_vars(
    distrib,
    init_params=None,
    num_class=None,
    random_seed=None,
):
    """Creates tfp.distribution and tf.Variables for the distribution's
    parameters.

    Parameters
    ----------
    num_class : int, optional
        only used when discrete random variable and number of classes is known.

    Returns
    -------
    (tfp.distribution.Distribution, dict('param': tf.Variables))
        the distribution and tf.Variables as its parameters.
    """
    if init_params is None:
        init_params = {}

    if distrib == 'dirichlet_multinomial':
        params = get_dirichlet_multinomial_param_vars(
            random_seed=random_seed,
            **init_params,
        )
        return (
            tfp.distributions.DirichletMultinomial(**params),
            params,
        )
    elif distrib == 'normal':
        params = get_normal_param_vars(random_seed=random_seed, **init_params)
        return (
            tfp.distributions.Normal(**params),
            params,
        )
    else:
        raise NotImplementedError(
            f'{distrib} is not a supported '
            + 'distribution for `get_param_vars()`.'
        )


def get_dirichlet_multinomial_param_vars(
    num_classes=None,
    max_concentration=None,
    max_total_count=None,
    total_count=None,
    concentration=None,
    random_seed=None,
    name='dirichlet_multinomial',
):
    """Create tf.Variable parameters for the Dirichlet distribution."""
    with tf.name_scope(name):
        if num_classes and max_concentration and max_total_count:
            return {
                'total_count': tf.Variable(
                    initial_value=np.random.uniform(
                        1,
                        max_total_count,
                        num_classes,
                    ),
                    dtype=tf.int32,
                    name='total_count',
                ),
                'concentration': tf.Variable(
                    initial_value=np.random.uniform(
                        1,
                        max_concentration,
                        num_classes,
                    ),
                    dtype=tf.float32,
                    name='concentration',
                ),
            }
        elif total_count and concentration:
            return {
                'total_count': tf.Variable(
                    initial_value=total_count,
                    dtype=tf.int32,
                    name='total_count',
                ),
                'concentration': tf.Variable(
                    initial_value=concentration,
                    dtype=tf.float32,
                    name='concentration',
                ),
            }
        else:
            raise ValueError('Must pass either both `total_count` and '
                + '`concentration` xor pass `num_classes`, `max_total_count` and '
                + '`max_concentration`'
            )


def get_normal_param_vars(
    mean,
    standard_deviation,
    random_seed=None,
    name='normal',
):
    """Create tf.Variable parameters for the normal distribution.

    Parameters
    ----------
    mean : float | dict
        either a float as the initial value of the mean, or a dict containing
        the mean and standard deviation of a normal distribution which this
        mean is drawn from randomly.
    standard_deviation : float | dict
        either a float as the initial value of the standard_deviation, or a dict
        containing the mean and standard deviation of a normal distribution
        which this mean is drawn from randomly.
    """
    with tf.name_scope(name):
        if isinstance(mean, dict) and isinstance(standard_deviation, dict):
            return {
                'loc': tf.Variable(
                    initial_value=np.random.normal(**mean),
                    dtype=tf.float32,
                    name='mean',
                ),
                'scale': tf.Variable(
                    initial_value=np.random.uniform(**standard_deviation),
                    dtype=tf.float32,
                    name='standard_deviation',
                ),
            }
        elif isinstance(mean, float) and isinstance(standard_deviation, float):
            return {
                'loc': tf.Variable(
                    initial_value=mean,
                    dtype=tf.float32,
                    name='mean',
                ),
                'scale': tf.Variable(
                    initial_value=standard_deviation,
                    dtype=tf.float32,
                    name='standard_deviation',
                ),
            }
        else:
            raise TypeError(
                'Both `mean` and `standard_deviation` must either be floats '
                + 'xor dicts containing a mean and standard_deviation each for sampling '
                + 'from a normal distribution to select the initial values. '
                + f'Not {type(mean)} and {type(standard_deviation)}'
            )

