"""The Tensorflow optimization of either a distribution or a Bayesian Neural
Network using MCMC methods.
"""
import logging

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distribution_tests import get_tfp_distrib
from psych_metric.distribution_tests import get_num_params
from psych_metric.distrib.mle_gradient_descent import get_distrib_param_vars
from psych_metric.distrib.mle_gradient_descent import MLEResults
from psych_metric.distrib.tfp_mvst import MultivariateStudentT

# TODO add tensorboard for visualizing and keeping track of progress.

def mcmc_distrib_params(
    distrib_id,
    data,
    params,
    const_params=None,
    kernel_id='NoUTurnSampler',
    kernel_args=None,
    step_adjust_args=None,
    num_top=1,
    num_samples=int(1e4),
    burnin=int(1e5),
    lag=int(1e4),
    parallel_iter=10,
    dtype=tf.float32,
    random_seed=None,
    alt_distrib=False,
    constraint_multiplier=1e5,
    sess_config=None,
):
    """Performs MCMC over the parameter distributions and returns the parameter
    set with the highest maxmimum likelihood estimate.
    """
    if kernel_args is None:
        # Ensure that opt args is a dict for use with **
        # TODO decide on default values for the kernel args
        if kernel_id == 'RandomWalkMetropolis':
            kernel_args = {'scale': 0.5}
        elif kernel_id == 'HamiltonianMonteCarlo':
            kernel_args = {
                'step_size': 0.001, # initial step size
                'num_leapfrog_steps': 2,
            }
        elif kernel_id == 'NoUTurnSampler':
            kernel_args = {
                'step_size': 0.1, # initial step size
                'num_leapfrog_steps': 2,
            }

        # create default step adjust args?
        if step_adjust_args is None:
            step_adjust_args = {
                'num_adaptation_steps': np.floor(burnin * 0.6),
            }
    if random_seed:
        # Set random seed if given.
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

    if const_params is None:
        const_params = {}
    elif isinstance(const_params, list):
        const_params = {k:v for k, v in params.items() if k in const_params}

    loss_fn = lambda x: tfp_log_prob(
        x,
        const_params,
        data,
        get_tfp_distrib(distrib_id),
        lambda y: unpack_mvst_params(
            y,
            data.shape[1],
            'df' not in const_params,
            'loc' not in const_params,
            'scale' not in const_params and 'sigma' not in const_params,
        ),
    )

    kernel = get_mcmc_kernel(loss_fn, kernel_id, kernel_args)

    current_state = pack_mvst_params(params, const_params)

    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_samples,
        current_state=current_state,
        kernel=kernel,
        num_burnin_steps=burnin,
        num_steps_between_results=lag,
        parallel_iterations=parallel_iter,
    )

    # TODO get maximum likelihood estimate parameter from parameter set
    log_probs = tf.map_fn(
        loss_fn,
        samples,
        parallel_iterations=parallel_iter,
        back_prop=False,
    )

    with tf.Session(config=sess_config) as sess:
        sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

        iter_results = sess.run({
            'log_probs': log_probs,
            'samples': samples,
            'trace': trace,
        })

    max_idx = np.argmax(iter_results['log_probs'])

    return iter_results, max_idx
    #return iter_results['log_probs'][max_idx], iter_results['samples'][max_idx]


def get_mcmc_kernel(loss_fn, kernel_id, kernel_args, step_adjust_args=None):
    # TODO setup tfp.mcmc.SimpleStepSizeAdaptation
    kernel_id = kernel_id.lower()

    if kernel_id == 'randomwalk' or kernel_id == 'randomwalkmetropolis':
        return tfp.mcmc.RandomWalkMetropolis(
            loss_fn,
            tfp.mcmc.random_walk_uniform_fn(kernel_args['scale']),
        )
    if kernel_id == 'nuts' or kernel_id == 'nouturnsampler':
        nuts = tfp.mcmc.NoUTurnSampler(loss_fn, **kernel_args)

        if step_adjust_args:
            return tfp.mcmc.SimpleStepSizeAdaptation(nuts, **step_adjust_args)
        return nuts
    if (
        kernel_id == 'hmc'
        or kernel_id == 'hmcmc'
        or kernel_id == 'hamiltonianmontecarlo'
    ):
        hmc = tfp.mcmc.HamiltonianMonteCarlo(loss_fn, **kernel_args)

        if step_adjust_args:
            return tfp.mcmc.SimpleStepSizeAdaptation(hmc, **step_adjust_args)
        return hmc

    raise ValueError(f'Unexpected value for `kernel_id`: {kernel_id}')


def unpack_mvst_params(params, dims, df=True, loc=True, scale=True):
    """Unpacks the parameters from a 1d-array."""
    if df and loc and scale:
        return {
            'df': params[0],
            'loc': params[1 : dims + 1],
            'scale': tf.reshape(params[dims + 1:], [dims, dims]),
        }

    if not df and loc and scale:
        return {
            'loc': params[:dims],
            'scale': tf.reshape(params[dims:], [dims, dims]),
        }

    if not df and not loc and scale:
        return {'scale': tf.reshape(params, [dims, dims])}

    if not df and loc and not scale:
        return {'loc': params}

    if df and not loc and scale:
        return {'df': params[0], 'scale': tf.reshape(params[1:], [dims, dims])}


def pack_mvst_params(params, const_params):
    """Packs the parameters into a 1d-array."""
    arr = []
    if 'df' not in const_params:
        arr.append([params['df']])

    if 'loc' not in const_params:
        arr.append(params['loc'])

    if 'scale' in params and 'scale' not in const_params:
        arr.append(params['scale'].flatten())
    elif 'sigma' in params and 'sigma' not in const_params:
        arr.append(params['sigma'].flatten())
    elif (
        'covariance_matrix' in params
        and 'covariance_matrix' not in const_params
    ):
        arr.append(params['covariance_matrix'].flatten())

    return np.concatenate(arr)


def tfp_log_prob(params, const_params, data, distrib_class, unpack_params):
    parameters = unpack_params(params)
    parameters.update(const_params)
    distrib = distrib_class(**parameters)

    # TODO handle parameter constraints (use get_mle_loss)
    log_prob, loss = get_mle_loss(
        data,
        distrib,
        parameters,
        const_params,
        neg_loss=False,
    )

    return tf.reduce_sum(loss, name='tfp_log_prob_sum')


def get_mle_loss(
    data,
    distrib,
    params,
    const_params,
    alt_distrib=False,
    constraint_multiplier=1e5,
    neg_loss=True,
):
    """Given a tfp distrib, create the MLE loss."""
    log_prob = distrib.log_prob(value=data)

    # Calc neg log likelihood to find minimum of (aka maximize log likelihood)
    if neg_loss:
        neg_log_prob = -1.0 * tf.reduce_sum(log_prob, name='neg_log_prob_sum')
    else:
        neg_log_prob = tf.reduce_sum(log_prob, name='neg_log_prob_sum')

    loss = neg_log_prob
    # Apply param constraints. Add Relu to enforce positive values
    if isinstance(distrib, tfp.distributions.Dirichlet) and alt_distrib:
        if const_params is None or 'precision' not in const_params:
            if neg_loss:
                loss = loss +  constraint_multiplier \
                    * tf.nn.relu(-params['precision'] + 1e-3)
            else:
                loss = loss - constraint_multiplier \
                    * tf.nn.relu(-params['precision'] + 1e-3)
    elif isinstance(distrib, tfp.distributions.MultivariateStudentTLinearOperator):
        if const_params is None or 'df' not in const_params:
            # If opt Student T, mean is const and Sigma got from df & cov
            # thus enforce df > 2, ow. cov is undefined.

            # NOTE cannot handle df < 2, ie. cannot learn Multivariate Cauchy

            # for df > 2 when given Covariance_matrix as a const parameter,
            # rather than scale
            if 'covariance_matrix' in const_params:
                if neg_loss:
                    loss = loss + constraint_multiplier \
                        * tf.nn.relu(-params['df'] + 2 + 1e-3)
                else:
                    loss = loss - constraint_multiplier \
                        * tf.nn.relu(-params['df'] + 2 + 1e-3)
            else:
                # enforce it to be greater than 0
                if neg_loss:
                    loss = loss + constraint_multiplier \
                        * tf.nn.relu(-params['df'] + 1e-3)
                else:
                    loss = loss - constraint_multiplier \
                        * tf.nn.relu(-params['df'] + 1e-3)

    return neg_log_prob, loss
