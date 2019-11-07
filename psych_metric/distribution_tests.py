"""Functions for performing distribution model selection."""
import functools
import logging
import math
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

@functools.total_ordering
class MLEResults(object):
    def _is_valid_operand(self, other):
        return (
            hasattr(other, "neg_log_likelihood")
            and hasattr(other, "params")
            and hasattr(other, "info_criterion")
        )

    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.neg_log_likelihood == other.neg_log_likelihood

    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.neg_log_likelihood < other.neg_log_likelihood

    def __init__(
        self,
        neg_log_likelihood: float,
        params=None,
        info_criterion=None,
    ):
        self.neg_log_likelihood = neg_log_likelihood
        self.params = params
        self.info_criterion = info_criterion


def aic(mle, num_params, log_mle=True):
    """Akaike information criterion (unadjusted).

    Parameters
    ----------
    mle : float
        Maximum Likelihood Estimate (MLE).
    num_params : int
        The number of parameters of the model whose MLE is given.
    log_mle : bool, optional
        True if the given `mle` is already the Log MLE, otherwise the log of
        the `mle` is taken in the equation.

    Returns
    -------
    float
        The Akaike information criterion of the given MLE for the model.
    """
    if log_mle:
        return 2 * (num_params - mle)
    return 2 * (num_params - math.log(mle))


def bic(mle, num_params, num_samples, log_mle=True):
    """Bayesian information criterion. Approx. Bayes Factor.

    Parameters
    ----------
    mle : float
        Maximum Likelihood Estimate (MLE).
    num_params : int
        The number of parameters of the model whose MLE is given.
    num_samples : int
        The number of samples used to fit the model whose MLE is given.
    log_mle : bool, optional
        True if the given `mle` is already the Log MLE, otherwise the log of
        the `mle` is taken in the equation.

    Returns
    -------
    float
        The Bayesian information criterion of the given MLE for the model.
    """
    if log_mle:
        return num_params * math.log(num_samples) - 2 * mle
    return num_params * math.log(num_samples) - 2 * math.log(mle)


def hqc(mle, num_params, num_samples, log_mle=True):
    """Hannan-Quinn information criterion.

    Parameters
    ----------
    mle : float
        Maximum Likelihood Estimate (MLE).
    num_params : int
        The number of parameters of the model whose MLE is given.
    num_samples : int
        The number of samples used to fit the model whose MLE is given.
    log_mle : bool, optional
        True if the given `mle` is already the Log MLE, otherwise the log of
        the `mle` is taken in the equation.

    Returns
    -------
    float
        The HWC of the given MLE for the model.
    """
    if log_mle:
        return 2 * (num_params * math.log(math.log(num_samples)) - mle)
    return 2 * (num_params * math.log(math.log(num_samples)) - math.log(mle))


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


def calc_info_criterion(mle, num_params, criterions, num_samples=None):
    """Calculate information criterions with mle_list and other information."""
    info_criterion = {}

    if 'bic' in criterions:
        info_criterion['bic'] = bic(mle, num_params, num_samples)

    if 'aic' in criterions:
        info_criterion['aic'] = aic(mle, num_params)

    if 'hqc' in criterions:
        info_criterion['hqc'] = hqc(mle, num_params, num_samples)

    return info_criterion


# NOTE MLE search over params  could be done in SHADHO instead
def mle_adam(
    distrib_id,
    data,
    init_params=None,
    const_params=None,
    optimizer_args=None,
    num_top_likelihoods=1,
    max_iter=10000,
    tol_param=1e-8,
    tol_loss=1e-8,
    tol_grad=1e-8,
    tb_summary_dir=None,
    random_seed=None,
    shuffle_data=True,
    batch_size=1,
    name='MLE_adam',
    sess_config=None,
    tf_optimizer='adam',
    tol_chain=1,
    alt_distrib=False,
    constraint_multiplier=1e5,
):
    """Uses tensorflow's ADAM optimizer to search the parameter space for MLE.

    Parameters
    ----------
    distrib_id : str
        Name of the distribution whoe MLE is being found.
    data : np.ndarray
    init_params : dict, optional
        The initial parameters of the distribution. Otherwise, selected randomly.
    const_params : list(str), optional
        The names of the parameters to be kept constant, all others are assumed
        to be variables.
    optimizer_args : dict, optional
    num_top_likelihoods : int, optional
        The number of top best likelihoods and their respective parameters. If
        equal to or less than -1, returns full history of likelihoods.
    tb_summary_dir : str, optional
        directory path to save TensorBoard summaries.
    name : str
        Name prefixed to Ops created by this class.
    alt_distrib : bool
        whether to use the alternate version of parameterization of the given
        distribution.
    constraint_multiplier : float, optional
        The multiplier to use to enforce constraints on the params in the loss.
        Typically a large positive value.

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
        # tensorflow prep data
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if shuffle_data:
            dataset = dataset.shuffle(batch_size, seed=random_seed)
        dataset = dataset.repeat(max_iter)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        tf_data = tf.cast(iterator.get_next(), tf.float32)

        # create distribution and dict of the distribution's parameters
        distrib, params = get_distrib_param_vars(
            distrib_id,
            init_params,
            const_params,
            alt_distrib=alt_distrib,
        )

        #log_prob = distrib.log_prob(value=data)
        log_prob = distrib.log_prob(value=tf_data)

        # Calc neg log likelihood to find minimum of (aka maximize log likelihood)
        neg_log_likelihood = -1.0 * tf.reduce_sum(
            log_prob,
            name='neg_log_likelihood_sum',
        )

        # Add Relu to enforce positive values for param constraints:
        if distrib_id.lower() == 'dirichlet' and alt_distrib:
            if const_params is None or 'precision' not in const_params:
                loss = neg_log_likelihood +  constraint_multiplier \
                    * tf.nn.relu(-params['precision'] + 1e-3)
        else:
            loss = neg_log_likelihood

        # Create optimizer
        if tf_optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(**optimizer_args)
        elif tf_optimizer == 'nadam':
            optimizer = tf.contrib.opt.NadamOptimizer(**optimizer_args)
        else:
            raise ValueError(f'Unexpected tf_optimizer value: {tf_optimizer}')

        global_step = tf.Variable(0, name='global_step', trainable=False)

        if const_params:
            grad = optimizer.compute_gradients(
                loss,
                [v for k, v in params.items() if k not in const_params],
            )
        else:
            grad = optimizer.compute_gradients(
                loss,
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

        loss_chain = 0

        i = 1
        continue_loop = True
        while continue_loop:
            # get likelihood and params
            # TODO could remove const params from this for calc efficiency. Need to recognize those constants will be missing in returned params though.
            results_dict = {
                'train_op': train_op,
                'loss': loss,
                'neg_log_likelihood': neg_log_likelihood,
                #'log_prob': log_prob,
                #'tf_data': tf_data,
                'params': params,
                'grad': grad,
            }

            if tb_summary_dir:
                results_dict['summary_op'] = summary_op

            iter_results = sess.run(results_dict, {tf_data: data})

            for param, value in iter_results['params'].items():
                if np.isnan(value).any():
                    raise ValueError(f'{param} is NaN!')

            if iter_results['params']['precision'] > 0 and (not top_likelihoods or iter_results['loss'] < top_likelihoods[-1].neg_log_likelihood):
                # update top likelihoods and their respective params
                if num_top_likelihoods == 1:
                    top_likelihoods = [MLEResults(
                        iter_results['neg_log_likelihood'],
                        iter_results['params'],
                    )]
                elif num_top_likelihoods > 1:
                    # TODO Use better data structures for large num_top_likelihoods
                    top_likelihoods.append(MLEResults(
                        iter_results['neg_log_likelihood'],
                        iter_results['params'],
                    ))
                    top_likelihoods = sorted(top_likelihoods)

                    if len(top_likelihoods) > num_top_likelihoods:
                        del top_likelihoods[-1]

            if tb_summary_dir:
                # Write summary update
                summary_writer.add_summary(iter_results['summary_op'], i)
                summary_writer.flush()

            # Calculate Termination Conditions
            # TODO Does the values need sorted by keys first?
            if params_history:
                # NOTE beware possibility: hstack not generalizing, & may need squeeze()
                new_params = np.hstack(list(iter_results['params'].values()))
                prior_params = np.hstack(list(params_history[-1].values()))
                param_diff = np.subtract(new_params, prior_params)

                if np.linalg.norm(param_diff) < tol_param:
                    logging.info('Parameter convergence in %d iterations.', i)
                    continue_loop = False

            if loss_history and np.abs(iter_results['neg_log_likelihood'] - loss_history[-1]) < tol_param:
                loss_chain += 1

                if loss_chain >= tol_chain:
                    logging.info('Loss convergence in %d iterations.', i)
                    continue_loop = False
            else:
                #loss_chain = 0
                if loss_chain > 0:
                    loss_chain -= 1


            if loss_history and (np.linalg.norm(iter_results['grad']) < tol_param).all():
                logging.info('Gradient convergence in %d iterations.', i)
                continue_loop = False

            if i >= max_iter:
                logging.info('Maimum iterations (%d) reached without convergence.', max_iter)
                continue_loop = False

            # Save observed vars of interest
            if num_top_likelihoods <= 0 or not params_history and not loss_history:
                # return the history of all likelihoods and params.
                params_history.append(iter_results['params'])
                loss_history.append(iter_results['neg_log_likelihood'])
            else:
                # keep only last mle for calculating tolerance.
                params_history[0] = iter_results['params']
                loss_history[0] = iter_results['neg_log_likelihood']

            i += 1

    if tb_summary_dir:
        summary_writer.close()

    if num_top_likelihoods < 0:
        #return list(zip(loss_history, params_history))
        # NOTE beware that this will copy all of this history into memory before leaving scope.
        return [
            MLEResults(loss_history[i], params_history[i]) for i in
            range(len(loss_history))
        ]

    return top_likelihoods


def get_distrib_param_vars(
    distrib_id,
    init_params,
    const_params=None,
    num_class=None,
    random_seed=None,
    alt_distrib=False
):
    """Creates tfp.distribution and tf.Variables for the distribution's
    parameters.

    Parameters
    ----------
    distrib_id : str
        Name of the distribution being created.
    init_params : dict
        Initial parameters of the distribution.
    const_params : list(str), optional
        The names of the parameters to be kept constant, all others are assumed
        to be variables.
    num_class : int, optional
        only used when discrete random variable and number of classes is known.
    random_seed : int
        The random seed to use for initializing the distribution.

    Returns
    -------
    (tfp.distribution.Distribution, dict('param': tf.Variables))
        the distribution and tf.Variables as its parameters.
    """
    if distrib_id.lower() == 'dirichletmultinomial'or distrib_id == 'dirichlet_multinomial':
        params = get_dirichlet_multinomial_param_vars(
            random_seed=random_seed,
            const_params=const_params,
            **init_params,
        )
        return (
            tfp.distributions.DirichletMultinomial(**params),
            params,
        )
    elif distrib_id.lower() == 'dirichlet' and not alt_distrib:
        params = get_dirichlet_param_vars(
            random_seed=random_seed,
            const_params=const_params,
            **init_params,
        )
        return (
            tfp.distributions.Dirichlet(**params),
            params,
        )
    elif distrib_id.lower() == 'dirichlet' and alt_distrib:
        if 'concentration' in init_params:
            precision = np.sum(init_params['concentration'])
            # mean needs to be the discrete probability distrib.
            if np.isclose(precision, 1):
                precision = len(init_params['concentration'])
                mean = init_params['concentration']
            elif precision > 1:
                # Make mean be the normalized mean so all elements within [0,1]
                mean = init_params['concentration'] / precision
            else:
                # If precision is less than 1, that's a problem
                # TODO normalize the mean when it sums to less than 1.
                raise ValueError(' '.join([
                    '`precision`, the sum of the values in `concentration`,',
                    'is less than 1. This means that `mean`\'s values will',
                    'need normalized to fit within the range [0,1] and the',
                    'precision will be set to the sum of the uniform vector',
                    'of ones, which is the number dimensions.',
                ]))

            params = get_dirichlet_alt_param_vars(
                random_seed=random_seed,
                const_params=const_params,
                mean=mean,
                precision=precision,
            )
        else:
            params = get_dirichlet_alt_param_vars(
                random_seed=random_seed,
                const_params=const_params,
                **init_params,
            )
        return (
            tfp.distributions.Dirichlet(params['mean'] * params['precision']),
            params,
        )
    elif distrib_id.lower() == 'normal':
        params = get_normal_param_vars(
            random_seed=random_seed,
            const_params=const_params,
            **init_params,
        )
        return (
            tfp.distributions.Normal(**params),
            params,
        )
    else:
        raise NotImplementedError(
            f'{distrib_id} is not a supported '
            + 'distribution for `get_param_vars()`.'
        )


def get_dirichlet_param_vars(
    num_classes=None,
    max_concentration=None,
    concentration=None,
    const_params=None,
    random_seed=None,
    name='dirichlet_params',
):
    """Create tf.Variable parameters for the Dirichlet distribution."""
    with tf.name_scope(name):
        if num_classes and max_concentration:
            return {
                'concentration': tf.Variable(
                    initial_value=np.random.uniform(
                        0,
                        max_concentration,
                        num_classes,
                    ),
                    dtype=tf.float32,
                    name='concentration',
                ),
            }
        elif concentration is not None:
            return {
                'concentration': tf.constant(
                    value=concentration,
                    dtype=tf.float32,
                    name='concentration',
                ) if const_params and 'concentration' in const_params else tf.Variable(
                    initial_value=concentration,
                    dtype=tf.float32,
                    name='concentration',
                ),
            }
        else:
            raise ValueError(' '.join([
                'Must pass either both and `concentration` xor pass',
                '`num_classes`, and `max_concentration`',
            ]))


def get_dirichlet_alt_param_vars(
    num_classes=None,
    mean=None,
    max_precision=None,
    precision=None,
    const_params=None,
    random_seed=None,
    name='dirichlet_mean_precision_params',
):
    """Create tf.Variable parameters for the Dirichlet distribution using mean
    and precision.
    """
    with tf.name_scope(name):
        if num_classes and max_precision:
            return {
                'mean': tf.Variable(
                    initial_value=np.random.uniform(0, 1, num_classes),
                    dtype=tf.float32,
                    name='mean',
                ),
                'precision': tf.Variable(
                    initial_value=np.random.uniform(
                        0,
                        max_precision,
                        num_classes,
                    ),
                    dtype=tf.float32,
                    name='precision',
                ),
            }
        elif mean is not None and precision is not None:
            return {
                'mean': tf.constant(
                    value=mean,
                    dtype=tf.float32,
                    name='mean',
                ) if const_params and 'mean' in const_params else tf.Variable(
                    initial_value=mean,
                    dtype=tf.float32,
                    name='mean',
                ),
                'precision': tf.constant(
                    value=precision,
                    dtype=tf.float32,
                    name='precision',
                ) if const_params and 'precision' in const_params else tf.Variable(
                    initial_value=precision,
                    dtype=tf.float32,
                    name='precision',
                ),
            }
        else:
            raise ValueError(' '.join([
                'Must pass either both `mean` and `precision` xor pass',
                '`num_classes`, and `max_precision`',
            ]))


def get_dirichlet_multinomial_param_vars(
    num_classes=None,
    max_concentration=None,
    max_total_count=None,
    total_count=None,
    concentration=None,
    const_params=None,
    random_seed=None,
    name='dirichlet_multinomial_params',
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
                    dtype=tf.float32,
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
        elif total_count is not None and concentration is not None:
            return {
                'total_count': tf.constant(
                    value=total_count,
                    dtype=tf.float32,
                    name='total_count',
                ) if const_params and 'total_count' in const_params else tf.Variable(
                    initial_value=total_count,
                    dtype=tf.float32,
                    name='total_count',
                ),
                'concentration': tf.constant(
                    value=concentration,
                    dtype=tf.float32,
                    name='concentration',
                ) if const_params and 'concentration' in const_params else tf.Variable(
                    initial_value=concentration,
                    dtype=tf.float32,
                    name='concentration',
                ),
            }
        else:
            raise ValueError('Must pass either both `total_count` and '
                + '`concentration` xor pass `num_classes`, `max_total_count` '
                + 'and `max_concentration`'
            )


def get_normal_param_vars(
    loc,
    scale,
    random_seed=None,
    const_params=None,
    name='normal_params',
):
    """Create tf.Variable parameters for the normal distribution.

    Parameters
    ----------
    loc : float | dict
        either a float as the initial value of the loc, or a dict containing
        the loc and standard deviation of a normal distribution which this
        loc is drawn from randomly.
    scale : float | dict
        either a float as the initial value of the scale, or a dict
        containing the loc and standard deviation of a normal distribution
        which this loc is drawn from randomly.
    """
    with tf.name_scope(name):
        params = {}

        # Get loc
        if isinstance(loc, dict):
            params['loc'] = (tf.constant(
                value=np.random.normal(**loc),
                dtype=tf.float32,
                name='loc',
            ) if const_params and 'loc' in const_params else tf.Variable(
                initial_value=np.random.normal(**loc),
                dtype=tf.float32,
                name='loc',
            ))
        elif isinstance(loc, float):
            params['loc'] = (tf.constant(
                value=loc,
                dtype=tf.float32,
                name='loc',
            ) if const_params and 'loc' in const_params else tf.Variable(
                initial_value=loc,
                dtype=tf.float32,
                name='loc',
            ))
        else:
            raise TypeError(
                '`loc` must be either a float xor dict containing a loc and '
                + 'scale for sampling from a normal distribution to select the '
                + f'initial values. But recieved type: {type(loc)}'
            )

        # Get scale
        if isinstance(scale, dict):
            params['scale'] = (tf.constant(
                value=np.random.normal(**scale),
                dtype=tf.float32,
                name='scale',
            ) if const_params and 'scale' in const_params else tf.Variable(
                initial_value=np.random.normal(**scale),
                dtype=tf.float32,
                name='scale',
            ))
        elif isinstance(scale, float):
            params['scale'] = (tf.constant(
                value=scale,
                dtype=tf.float32,
                name='scale',
            ) if const_params and 'scale' in const_params else tf.Variable(
                initial_value=scale,
                dtype=tf.float32,
                name='scale',
            ))
        else:
            raise TypeError(
                '`scale` must be either a float xor dict containing a scale and '
                + 'scale for sampling from a normal distribution to select the '
                + f'initial values. But recieved type: {type(scale)}'
            )

        return params
