import math

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def AIC(mle, num_params):
    """Akaike information criterion."""
    return 2 * (num_params - math.log(mle))


def BIC(mle, num_params, num_samples):
    """Bayesian information criterion. Approx. Bayes Factor"""
    return num_params * math.log(num_samples) - 2 * math.log(mle)


def HQC(mle, num_params, num_samples):
    """Hannan-Quinn information criterion."""
    return 2 * (num_params * math.log(math.log(num_samples)) - mle)


def DIC(likelihood_function, num_params, num_samples, mle=None, mean_lf=None):
    """Deviance information criterion.

    Parameters
    ----------
    mle : float
        the Maximum Likelihood Estimate to repreesnt the distribution's
        likelihood function.
    """
    raise NotImplementedError("Need to implement finding the MLE from Likelihood function, and the expected value of the likelihood function.")

    if mean_lf is None:
        raise NotImplementedError("Need to implement finding the expected value from Likelihood function.")
    if mle is None:
        raise NotImplementedError("Need to implement finding the MLE from Likelihood function.")

    # DIC = 2 * (pd - D(theta)), where pd is spiegelhalters: pd= E(D(theta)) - D(E(theta))
    return 2 * (expected_value_likelihood_func - math.log(mle) - math.log(likelihood_funciton))

# TODO exhaustive likelihood calculation for distrib and parameter set and data.
def exhaustive_likelihood(distribution, data, optimizer_args={}):
    """Computes the likelihood using every data sample.

    Parameters
    ----------
    distribtion : tfp.distribution
        An initialized tfp distribution with preset
    data : np.ndarray
        The data whose source distribution is in quesiton.

    Returns
    -------
    float
        The likelihood of the distribution with that set of parameters being
        the source from which the data was drawn from.
    """

    return likelihood

# TODO MLE search over params (this could be done in SHADHO instead)
def mle_adam(
    distribution,
    data,
    optimizer_args=None,
    num_top_likelihoods=1,
    params=None,
    name='MLE_adam'
)
    """Uses tensorflow's ADAM optimizer to search the parameter space for MLE.

    Parameters
    ----------
    distribution : tfp.distribution
    data : np.ndarray
    optimizer_args : dict, optional
    num_top_likelihoods : int, optional
        The number of top best likelihoods and their respective parameters.
    params : dict, optional
        The initial parameters of the distribution. Otherwise, selected randomly.
    tb_summary_dir : str
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
        optimier_args = {}

    # tensor to hold the data
    with tf.name_scope(name) as scope:
        if isinstance(data, np.ndarray):
            data = tf.placeholder(dtype=tf.float32, name='data')

        if params is None:
            # create dict of the distribution's parameters
            params = get_param_vars(distribution)

        # TODO why negative? is this necessary? should it be a user passed flag?
        # because ths is the minimized loss, and we want the Maximumg Likelihood
        neg_log_likelihood = -1.0 * tf.reduce_sum(
            distribution.log_prob(value=data),
            name='log_likelihood_sum',
        )

        # Create optimizer
        optimizer = tf.train.AdamOptimizer(**optimizer_args)
        global_step = tf.Variable(0, name='global_step', trainable=False)

        gradients = optimizer.compute_gradients(neg_log_likelihood)
        train_op = optimizer.apply_gradients(gradients, global_step)

        # TODO summary ops?
        if tb_summary_dir:
            # Visualize the gradients
            for i, g in enumerate(gradients):
                tf.summary.histogram(f'{g[1].name}-grad', g[0])

            summary_op = tf.summary.merge_all()

    with tf.Session(config=sess_config) as sess:
        # Build summary operation
        if tb_summary_dir:
            summary_writer = tf.summary.FileWriter(
                os.path.join(
                    output_dir,
                    'summaries',
                    summary_path,
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
        while finding_mle:
            # get likelihood and params
            iter_results = sess.run({
                'train_op': train_op,
                'neg_log_likelihood': neg_log_likelihood,
                'params': params,
                'summary_op': summary_op,
            })

            if iter_results['neg_log_likelihood'] < top_likelihoods[-1]:
                # update top likelihoods
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

            if tb_summary_dir:
                # Write summary update
                summary_writer.add_summary(train_results['summary_op'], count)
                summary_writer.flush()

    if tb_summary_dir:
        summary_writer.close()

    return top_likelihoods


def get_param_vars(
    distribution,
    init_params=None,
    random_seed=None,
    name=None,
    num_class=None,
):
    """Creates tf.Variables for the distribution's parameters

    Parameters
    ----------
    num_class : int, optional
        only used when discrete random variable and number of classes is known.
    """
    if init_params is None:
        init_params = {}

    if distribution == 'dirichlet_multinomial':
        return get_dirichlet_multinomial_param_vars(**init_params)
    elif distribution == 'normal':
        return get_normal_param_vars(**init_params)
    else:
        raise NotImplementedError(f'{distribution} is not a supported '
            + 'distribution for `get_param_vars()`.'
        )


def get_dirichlet_multinomial_param_vars(
    num_classes=None,
    max_concentration=None,
    max_total_counts=None,
    total_counts=None,
    concentration=None,
)
    """Create tf.Variable parameters for the Dirichlet distribution."""
    if num_classes and max_concentration and max_total_counts:
        return {
            'total_counts': tf.Variable(
                initial_value=np.random.uniform(
                    1,
                    max_total_counts,
                    num_classes,
                ),
                dtype=tf.float32,
            ),
            'concentration': tf.Variable(
                initial_value=np.random.uniform(
                    1,
                    max_concentration,
                    num_classes,
                ),
                dtype=tf.float32,
            ),
        }
    elif total_counts and concentration:
        return {
            'total_counts': tf.Variable(
                initial_value=total_counts,
                dtype=tf.float32,
            ),
            'concentration': tf.Variable(
                initial_value=concentration,
                dtype=tf.float32,
            ),
        }
    else:
        raise ValueError('Must pass either both `total_counts` and '
            + '`concentration` xor pass `num_classes`, `max_total_counts` and '
            + '`max_concentration`'
        )


def get_normal_param_vars(mean, variance)
    """Create tf.Variable parameters for the normal distribution.

    Parameters
    ----------
    mean : float | dict
        either a float as the initial value of the mean, or a dict containing
        the mean and standard deviation of a normal distribution which this
        mean is drawn from randomly.
    variance : float | dict
        either a float as the initial value of the variance, or a dict
        containing the mean and standard deviation of a normal distribution
        which this mean is drawn from randomly.
    """
    if isinstance(mean, dict) and isinstance(variance, dict):
        return {
            'mean': tf.Variable(
                initial_value=np.random.normal(**mean),
                dtype=tf.float32,
            ),
            'variance': tf.Variable(
                initial_value=np.random.uniform(**variance),
                dtype=tf.float32,
            ),
        }
    elif isinstance(mean, float) and isinstance(variance, float):
        return {
            'total_counts': tf.Variable(
                initial_value=np.random.uniform(1, max_total_counts, num_classes),
                dtype=tf.float32,
            ),
            'concentration': tf.Variable(
                initial_value=np.random.uniform(1, max_concentration, num_classes),
                dtype=tf.float32,
            ),
        }
    else
        raise TypeError('Both `mean` and `variance` must either be floats xor '
            + 'dicts containing a mean and variance each for sampling from a '
            + 'normal distribution to select the initial values. '
            + f'Not {type(mean)} and {type(variance)}'
        )
