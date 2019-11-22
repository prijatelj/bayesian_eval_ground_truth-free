"""The Tensorflow optimization of either a distribution or a Bayesian Neural
Network using MCMC methods.
"""
import logging

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distribution_tests import get_tfp_distrib_params
from psych_metric.distribution_tests import get_num_params
from psych_metric.distrib.mle_gradient_descent import get_distrib_param_vars
from psych_metric.distrib.mle_gradient_descent import MLEResults
from psych_metric.distrib.tfp_mvst import MultivariateStudentT


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
            kernel_args = {'scale': 1.0}
        elif kernel_id == 'HamiltonianMonteCarlo':
            kernel_args = {
                'step_size': 0.1, # initial step size
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
                'num_adaptation_steps': np.floor(burnin * .6),
            }
    if random_seed:
        # Set random seed if given.
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
    if const_params is None:
        const_params = {}

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

    # TODO Need to figure out how to update the parameters via this method...
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

    return iter_results['log_probs'][max_idx], iter_results['samples'][max_idx]


def get_tfp_distrib(distrib_id):
    distrib_id = distrib_id.lower()
    if distrib_id == 'dirichlet':
        return tfp.distributions.Dirichlet
    if distrib_id == 'multivariatenormal' or distrib_id == 'mvn':
        return tfp.distributions.MultivariateNormalFullCovariance
    if distrib_id == 'multivariatestudentt' or distrib_id == 'mvst':
        return MultivariateStudentT


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
        # Returning tuple to stay consistent w/ other
        return {'scale': tf.reshape(params, [dims, dims])}

def pack_mvst_params(params, const_params):
    """Unpacks the parameters from a 1d-array."""
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
    # TODO handle parameter constraints
    return tf.reduce_sum(distrib.log_prob(data), name='tfp_log_prob_sum')


def get_mle_loss(
    data,
    distrib,
    params,
    const_params,
    alt_distrib=False,
    constraint_multiplier=1e5,
):
    """Given a tfp distrib, create the MLE loss."""
    log_prob = distrib.log_prob(value=data)

    # Calc neg log likelihood to find minimum of (aka maximize log likelihood)
    neg_log_prob = -1.0 * tf.reduce_sum(log_prob, name='neg_log_prob_sum')

    loss = neg_log_prob
    # Apply param constraints. Add Relu to enforce positive values
    if isinstance(distrib, tfp.distributions.Dirichlet) and alt_distrib:
        if const_params is None or 'precision' not in const_params:
            loss = loss +  constraint_multiplier \
                * tf.nn.relu(-params['precision'] + 1e-3)
    elif isinstance(distrib, tfp.distributions.MultivariateStudentTLinearOperator):
        if const_params is None or 'df' not in const_params:
            # If opt Student T, mean is const and Sigma got from df & cov
            # thus enforce df > 2, ow. cov is undefined.

            # NOTE cannot handle df < 2, ie. cannot learn Multivariate Cauchy

            # for df > 2 when given Covariance_matrix as a const parameter,
            # rather than scale
            if 'covariance_matrix' in const_params:
                loss = loss + constraint_multiplier \
                    * tf.nn.relu(-params['df'] + 2 + 1e-3)
            else:
                # enforce it to be greater than 0
                loss = loss + constraint_multiplier \
                    * tf.nn.relu(-params['df'] + 1e-3)

    return neg_log_prob, loss


def run_session(
    distrib_id,
    results_dict,
    tf_data,
    data,
    params,
    const_params,
    num_top=1,
    max_iter=int(1e4),
    sess_config=None,
):
    with tf.Session(config=sess_config) as sess:
        sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

        top_likelihoods = []

        params_history = []
        loss_history = []
        loss_chain = 0

        i = 1
        continue_loop = True
        while continue_loop:
            iter_results = sess.run(results_dict, {tf_data: data})

            # Check if any of the params are NaN. Fail if so.
            for param, value in iter_results['params'].items():
                if np.isnan(value).any():
                    raise ValueError(f'{param} is NaN!')

            if is_param_constraint_broken(params, const_params):
                # This still counts as an iteration, just nothing to save.
                if i >= max_iter:
                    logging.info(
                        'Maimum iterations (%d) reached without convergence.',
                        max_iter,
                    )
                    continue_loop = False

                i += 1
                continue

            top_likelihoods = update_top_likelihoods(
                top_likelihoods,
                iter_results['neg_log_prob'],
                params,
                num_top,
            )

            # Save observed vars of interest
            if num_top <= 0 or not params_history and not loss_history:
                # return the history of all likelihoods and params.
                params_history.append(iter_results['params'])
                loss_history.append(iter_results['neg_log_prob'])
            else:
                # keep only last mle for calculating tolerance.
                params_history[0] = iter_results['params']
                loss_history[0] = iter_results['neg_log_prob']

            conitnue_loop = to_continue(
                distrib_id,
                iter_results['neg_log_prob'],
                iter_results['params'],
                params_history,
                loss_history,
                i,
                max_iter,
            )

            i += 1

    if num_top < 0:
        #return list(zip(loss_history, params_history))
        # NOTE beware that this will copy all of this history into memory before leaving scope.
        return [
            MLEResults(loss_history[i], params_history[i]) for i in
            range(len(loss_history))
        ]

    return top_likelihoods


def is_param_constraint_broken(params, const_params):
    # param value check if breaks constraints: Skip to next if so.
    return (
        ('precision' in params and params['precision'] <= 0)
        or (
            'df' in params
            and (
                ('covariance_matrix' in params and params['df'] <= 2)
                or params['df'] <= 0
            )
        )
    )


def update_top_likelihoods(top_likelihoods, neg_log_prob, params, num_top=1):
    # Assess if necessary to save the valid likelihoods
    if not top_likelihoods or neg_log_prob < top_likelihoods[-1].neg_log_likelihood:
        # update top likelihoods and their respective params
        if num_top == 1:
            top_likelihoods = [MLEResults(neg_log_prob, params)]
        elif num_top > 1:
            # TODO Use better data structures for large num_top_prob
            top_likelihoods.append(MLEResults(neg_log_prob, params))
            top_likelihoods = sorted(top_likelihoods)

            if len(top_likelihoods) > num_top:
                del top_likelihoods[-1]

    return top_likelihoods


def to_continue(
    distrib_id,
    neg_log_prob,
    params,
    params_history,
    loss_history,
    loss_chain,
    iter_num,
    max_iter=1e4,
    grad=None,
    tol_param=1e-8,
    tol_loss=1e-8,
    tol_grad=1e-8,
    tol_chain=3,
):
    """Calculate Termination Conditions"""
    # Calculate parameter difference
    if params_history:
        # NOTE beware possibility: hstack not generalizing, & may need squeeze()
        #new_params = np.hstack(list(iter_results['params'].values()))

        if distrib_id == 'MultivariateStudentT':
            new_params = np.hstack([v for k, v in params.items() if k != 'sigma'])
            new_params = np.hstack([new_params, params['sigma'].flatten()])

            prior_params = np.hstack([v for k, v in params_history[-1].items() if k != 'sigma'])
            prior_params = np.hstack([prior_params, params_history[-1]['sigma'].flatten()])

        else:
            new_params = np.hstack(list(params.values()))
            prior_params = np.hstack(list(params_history[-1].values()))

        param_diff = np.subtract(new_params, prior_params)

        if np.linalg.norm(param_diff) < tol_param:
            logging.info('Parameter convergence in %d iterations.', iter_num)
            return False

    # Calculate loss difference
    if loss_history and np.abs(neg_log_prob - loss_history[-1]) < tol_param:
        loss_chain += 1

        if loss_chain >= tol_chain:
            logging.info('Loss convergence in %d iterations.', iter_num)
            return False
    else:
        if loss_chain > 0:
            loss_chain -= 1

    # Calculate gradient difference
    if (
        grad is not None
        and loss_history
        and (np.linalg.norm(grad) < tol_param).all()
    ):
        logging.info('Gradient convergence in %d iterations.', iter_num)
        return False

    # Check if at or over maximum iterations
    if iter_num >= max_iter:
        logging.info('Maimum iterations (%d) reached without convergence.', max_iter)
        return False

    return True
