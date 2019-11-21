"""Functions for performing distribution model selection and helper
functions.
"""
import functools
import logging
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.mvst import MultivariateStudentT

# TODO replace any tensroflow usage of MVST w/ the actual tfp version
from psych_metric.distrib.tf_nelder_mead_mvst import mvst_tf_log_prob

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
    optimizer_id='adam',
    tol_chain=1,
    alt_distrib=False,
    constraint_multiplier=1e5,
    dtype=tf.float32,
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
        # Ensure that opt args is a dict for use with **
        optimizer_args = {}
    if random_seed:
        # Set random seed if given.
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

    # tensor to hold the data
    with tf.name_scope(name) as scope:
        # tensorflow prep data
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if shuffle_data:
            # NOTE batchsize is default to 1, is this acceptable?
            dataset = dataset.shuffle(batch_size, seed=random_seed)
        dataset = dataset.repeat(max_iter)
        # TODO No batching of this process... Decide if batching is okay
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        tf_data = tf.cast(iterator.get_next(), dtype)

        # create distribution and dict of the distribution's parameters
        distrib, params = get_distrib_param_vars(
            distrib_id,
            init_params,
            const_params,
            alt_distrib=alt_distrib,
        )

        # Get the log prob of data when distrib has the existing.
        if distrib_id.lower() == 'multivariatestudentt':
            log_prob = mvst_tf_log_prob(tf_data, **params)
        else:
            #log_prob = distrib.log_prob(value=data)
            log_prob = distrib.log_prob(value=tf_data)

        # Calc neg log likelihood to find minimum of (aka maximize log likelihood)
        neg_log_likelihood = -1.0 * tf.reduce_sum(
            log_prob,
            name='neg_log_likelihood_sum',
        )

        # Apply param constraints. Add Relu to enforce positive values
        if distrib_id.lower() == 'dirichlet' and alt_distrib:
            if const_params is None or 'precision' not in const_params:
                loss = neg_log_likelihood +  constraint_multiplier \
                    * tf.nn.relu(-params['precision'] + 1e-3)
        elif distrib_id.lower() == 'multivariatestudentt':
            if const_params is None or 'df' not in const_params:
                # If opt Student T, mean is const and Sigma got from df & cov
                # thus enforce df > 2, ow. cov is undefined.

                # NOTE cannot handle df < 2, ie. cannot learn Multivariate Cauchy
                loss = neg_log_likelihood +  constraint_multiplier \
                    * tf.nn.relu(-params['df'] + 2 + 1e-3)
        else:
            loss = neg_log_likelihood

        # Create optimizer
        if optimizer_id == 'adam':
            optimizer = tf.train.AdamOptimizer(**optimizer_args)
        elif optimizer_id == 'nadam':
            optimizer = tf.contrib.opt.NadamOptimizer(**optimizer_args)
        else:
            raise ValueError(f'Unexpected optimizer_id value: {optimizer_id}')

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

            if tb_summary_dir:
                # Write summary update
                summary_writer.add_summary(iter_results['summary_op'], i)
                summary_writer.flush()

            # Check if any of the params are NaN. Fail if so.
            for param, value in iter_results['params'].items():
                if np.isnan(value).any():
                    raise ValueError(f'{param} is NaN!')

            # param value check if breaks constraints: Skip to next if so.
            if (
                'precision' in iter_results['params']
                and iter_results['params']['precision'] <= 0
                ) or (
                    'df' in iter_results['params']
                    and iter_results['params']['df'] <= 2
            ):
                # This still counts as an iteration, just nothing to save.
                if i >= max_iter:
                    logging.info(
                        'Maimum iterations (%d) reached without convergence.',
                        max_iter,
                    )
                    continue_loop = False

                i += 1
                continue

            # Assess if necessary to save the valid likelihoods
            if not top_likelihoods or iter_results['loss'] < top_likelihoods[-1].neg_log_likelihood:
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

            # Calculate Termination Conditions
            # TODO Does the values need sorted by keys first?
            if params_history:
                # NOTE beware possibility: hstack not generalizing, & may need squeeze()
                #new_params = np.hstack(list(iter_results['params'].values()))

                if distrib_id == 'MultivariateStudentT':
                    new_params = np.hstack([v for k, v in iter_results['params'].items() if k != 'sigma'])
                    new_params = np.hstack([new_params, iter_results['params']['sigma'].flatten()])

                    prior_params = np.hstack([v for k, v in params_history[-1].items() if k != 'sigma'])
                    prior_params = np.hstack([prior_params, params_history[-1]['sigma'].flatten()])

                else:
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
    distrib_id = distrib_id.lower()

    if distrib_id == 'dirichletmultinomial'or distrib_id == 'dirichlet_multinomial':
        params = get_dirichlet_multinomial_param_vars(
            random_seed=random_seed,
            const_params=const_params,
            **init_params,
        )
        return (
            tfp.distributions.DirichletMultinomial(**params),
            params,
        )
    elif distrib_id == 'dirichlet' and not alt_distrib:
        params = get_dirichlet_param_vars(
            random_seed=random_seed,
            const_params=const_params,
            **init_params,
        )
        return (
            tfp.distributions.Dirichlet(**params),
            params,
        )
    elif distrib_id == 'dirichlet' and alt_distrib:
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
    elif distrib_id == 'normal':
        params = get_normal_param_vars(
            random_seed=random_seed,
            const_params=const_params,
            **init_params,
        )
        return (
            tfp.distributions.Normal(**params),
            params,
        )
    elif distrib_id == 'multivariatestudentt':
        params = get_mvst_param_vars(
            const_params=const_params,
            **init_params,
        )
        return (
            #tfp.distributions.MultivariateStudentTLinearOperator(**params),
            None,
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
    dtype=tf.float32,
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
                    dtype=dtype,
                    name='concentration',
                ),
            }
        elif concentration is not None:
            return {
                'concentration': tf.constant(
                    value=concentration,
                    dtype=dtype,
                    name='concentration',
                ) if const_params and 'concentration' in const_params else tf.Variable(
                    initial_value=concentration,
                    dtype=dtype,
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
    dtype=tf.float32,
):
    """Create tf.Variable parameters for the Dirichlet distribution using mean
    and precision.
    """
    with tf.name_scope(name):
        if num_classes and max_precision:
            return {
                'mean': tf.Variable(
                    initial_value=np.random.uniform(0, 1, num_classes),
                    dtype=dtype,
                    name='mean',
                ),
                'precision': tf.Variable(
                    initial_value=np.random.uniform(
                        0,
                        max_precision,
                        num_classes,
                    ),
                    dtype=dtype,
                    name='precision',
                ),
            }
        elif mean is not None and precision is not None:
            return {
                'mean': tf.constant(
                    value=mean,
                    dtype=dtype,
                    name='mean',
                ) if const_params and 'mean' in const_params else tf.Variable(
                    initial_value=mean,
                    dtype=dtype,
                    name='mean',
                ),
                'precision': tf.constant(
                    value=precision,
                    dtype=dtype,
                    name='precision',
                ) if const_params and 'precision' in const_params else tf.Variable(
                    initial_value=precision,
                    dtype=dtype,
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
    dtype=tf.float32,
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
                    dtype=dtype,
                    name='total_count',
                ),
                'concentration': tf.Variable(
                    initial_value=np.random.uniform(
                        1,
                        max_concentration,
                        num_classes,
                    ),
                    dtype=dtype,
                    name='concentration',
                ),
            }
        elif total_count is not None and concentration is not None:
            return {
                'total_count': tf.constant(
                    value=total_count,
                    dtype=dtype,
                    name='total_count',
                ) if const_params and 'total_count' in const_params else tf.Variable(
                    initial_value=total_count,
                    dtype=dtype,
                    name='total_count',
                ),
                'concentration': tf.constant(
                    value=concentration,
                    dtype=dtype,
                    name='concentration',
                ) if const_params and 'concentration' in const_params else tf.Variable(
                    initial_value=concentration,
                    dtype=dtype,
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
    dtype=tf.float32,
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
                dtype=dtype,
                name='loc',
            ) if const_params and 'loc' in const_params else tf.Variable(
                initial_value=np.random.normal(**loc),
                dtype=dtype,
                name='loc',
            ))
        elif isinstance(loc, float):
            params['loc'] = (tf.constant(
                value=loc,
                dtype=dtype,
                name='loc',
            ) if const_params and 'loc' in const_params else tf.Variable(
                initial_value=loc,
                dtype=dtype,
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
                dtype=dtype,
                name='scale',
            ) if const_params and 'scale' in const_params else tf.Variable(
                initial_value=np.random.normal(**scale),
                dtype=dtype,
                name='scale',
            ))
        elif isinstance(scale, float):
            params['scale'] = (tf.constant(
                value=scale,
                dtype=dtype,
                name='scale',
            ) if const_params and 'scale' in const_params else tf.Variable(
                initial_value=scale,
                dtype=dtype,
                name='scale',
            ))
        else:
            raise TypeError(
                '`scale` must be either a float xor dict containing a scale and '
                + 'scale for sampling from a normal distribution to select the '
                + f'initial values. But recieved type: {type(scale)}'
            )

        return params


def get_mvst_param_vars(
    df,
    loc,
    covariance_matrix,
    const_params=None,
    name='multivariate_student_t_params',
    dtype=tf.float32,
):
    """Create tf.Variable parameters for the Multivariate Student distribution.

    Parameters
    ----------
    df : float
        Positive non-zero float for degrees of freedom.
    loc : np.ndarray(float)
        Vector of floats for the locations or means.
    scale : np.ndarray(floats), optional
        Symmetric positive definite matrix of floats for Sigma matrix.
        Expects Covariance Matrix that is held constant and is used along with
        df to calculate the scale.
    covariance_matrix : np.ndarray(floats), optional
        Symmetric positive definite matrix of floats for covariance matrix.
        Expects Covariance Matrix that is held constant and is used along with
        df to calculate the scale.
    """

    """
    raise NotImplementedError(' '.join([
        'Cannot do this as is. Cov = Sigma*df/(df-2). Sigma = scale @ scale.T',
        'However, this is only because the',
        'tfp.MultivariateStudentTLinearOperator uses `scale`. If it were to',
        'use Sigma, this would be doable via the MLE method estimating only',
        '`df`.',
    ]))
    #"""

    with tf.name_scope(name):
        params = {}

        # Get df
        if isinstance(df, float) and df > 2:
            params['df'] = (tf.constant(
                value=df,
                dtype=dtype,
                name='df',
            ) if const_params and 'df' in const_params else tf.Variable(
                initial_value=df,
                dtype=dtype,
                name='df',
            ))
        else:
            raise TypeError(
                '`df` must be either a positve float greater than 2.'
                + f'But recieved type: {type(df)}'
            )

        # Get loc
        if isinstance(loc, float) or isinstance(loc, list) or isinstance(loc, np.ndarray):
            params['loc'] = (tf.constant(
                value=loc,
                dtype=dtype,
                name='loc',
            ) if const_params and 'loc' in const_params else tf.Variable(
                initial_value=loc,
                dtype=dtype,
                name='loc',
            ))
        else:
            raise TypeError(
                '`loc` must be either a float xor vector of floats '
                + f'initial values. But recieved type: {type(loc)}'
            )

        # Get Sigma matrix
        # TODO make scale dependent on df and data covariance:
        # sigma = (df-2) / df * cov(data) when df > 2.
        cov = tf.constant(
            value=covariance_matrix,
            dtype=dtype,
            name='covariance_matrix',
        )

        #params['sigma'] = tf.linalg.LinearOperatorLowerTriangular(
        params['sigma'] = (df - 2) / df * cov

        return params
