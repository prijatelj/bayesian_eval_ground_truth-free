"""The Tensorflow optimization of either a distribution or a Bayesian Neural
Network using MCMC methods.
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distribution_tests import get_tfp_distrib_params
from psych_metric.distrib.mle_gradient_descent import get_distrib_param_vars
from psych_metric.distrib.mle_gradient_descent import MLEResults


def mcmc_distrib(
    distrib_id,
    data,
    init_params=None,
    const_params=None,
    kernel_id='nuts',
    kernel_args=None,
    num_top=1,
    num_samples=int(1e4),
    burnin=int(1e4),
    lag=int(1e3),
    parallel_iter=16,
    dtype=tf.float32,
    random_seed=None,
    alt_distrib=False,
    constraint_multiplier=1e5,
    sess_config=None,
    name='MCMC_distrib_params',
):
    if kernel_args is None:
        # Ensure that opt args is a dict for use with **
        # TODO decide on default values for the kernel args
        kernel_args = {
            'step_size': 0.1 # initial step size
            'num_leapfrog_steps': 2
        }
    if random_seed:
        # Set random seed if given.
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

    with tf.name_scope(name) as scope:
        distrib, params = get_distrib_param_vars(
            distrib_id,
            init_params,
            const_params,
            data.shape[1],
        )

        # TODO Need to give the kernels the actual loss function, not its results
        neg_log_prob, loss = get_mle_loss(
            data,
            distrib,
            params,
            const_params,
            alt_distrib,
            constraint_multiplier,
        )

        # TODO Need to figure out how to update the parameters via this method...
        kernel = get_mcmc_kernel(loss, kernel_id, kernel_args)

        samples, _ = tfp.mcmc.sample_chain(
            num_results=num_samples, # ? in this context for the distribs?
            current_state=distrib,
            kernel=kernel,
            num_burnin_steps=burnin,
            num_steps_between_results=lag,
            parallel_iterations=parallel_iter,
        )

    # TODO run session to get the parameters
    results_dict = {
        'samples': samples,
        'params': params,
        'neg_log_prob': neg_log_prob,
        #'loss': loss,
    }

    mle_params = run_session(results_dict, num_top, sess_config)

    # TODO Should this return the distribution over the params or just param
    # set w/ Maxmimum Likelihoo Estimation?

    return mle_params


def get_mcmc_kernel(loss, kernel_id, kernel_args):
    # TODO setup tfp.mcmc.SimpleStepSizeAdaptation
    kernel_id = kernel_id.lower()

    if kernel_id == 'randomwalk' or kernel_id == 'randomwalkmetropolis':
        return tfp.mcmc.RandomWalkMetropolis(loss, **kernel_args)
    if kernel_id == 'nuts' or kernel_id == 'nouturnsampler':
        return tfp.mcmc.NoUTurnSampler(loss, **kernel_args)
    if (
        kernel_id == 'hmc'
        or kernel_id == 'hmcmc'
        or kernel_id == 'HamiltonianMonteCarlo'
    ):
        return tfp.mcmc.HamiltonianMonteCarlo(**kernel_args)

    raise ValueError(f'Unexpected value for `kernel_id`: {kernel_id}')


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
    results_dict,
    num_top=1,
    sess_config=None,
):
    with tf.Session(config=sess_config) as sess:
        sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

        top_likeilhoods = []

        param_history = []
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

            if is_param_constraint_broken(params):
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
                iter_results['neg_log_prob'],
                iter_results['params'],
                params_history,
                loss_history,
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


def top_likelihoods(top_likelihoods, neg_log_prob, params, num_top=1):
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
    neg_log_prob,
    params,
    params_history,
    loss_history,
    max_iter=1e4,
    grads=None,
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
            logging.info('Parameter convergence in %d iterations.', i)
            return False

    # Calculate loss difference
    if loss_history and np.abs(neg_log_prob - loss_history[-1]) < tol_param:
        loss_chain += 1

        if loss_chain >= tol_chain:
            logging.info('Loss convergence in %d iterations.', i)
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
        logging.info('Gradient convergence in %d iterations.', i)
        return False

    # Check if at or over maximum iterations
    if i >= max_iter:
        logging.info('Maimum iterations (%d) reached without convergence.', max_iter)
        return False

    return True
