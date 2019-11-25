"""The Tensorflow optimization of either a distribution or a Bayesian Neural
Network using MCMC methods.
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distrib.mle_gradient_descent import get_distrib_param_vars
from psych_metric.distrib.mle_gradient_descent import MLEResults
from psych_metric.distrib.tfp_mvst import MultivariateStudentT


# TODO this [un]pack step is unnecessary for mcmc, just need to know which
# params are not const, cuz can give a list of tensors to the current state
# have that be the state. thus, still requiring an unpacking function, but much
# simpler, cuz it just needs to know the order of params given.
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
