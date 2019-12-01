"""The Bayesian Neural Network that learns the transformation of the target
distribution to the predictor distribution, which results in the conditional
distribution of the predictor conditional on the target.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distrib.mcmc import get_mcmc_kernel


def bnn_mlp(
    input_labels,
    num_layers=2,
    num_hidden=10,
    hidden_activation=tf.math.sigmoid,
    hidden_use_bias=True,
    output_activation=tf.math.sigmoid,
    dtype=tf.float32,
    tf_device=None,
):
    """BNN of a simple MLP model. Input: labels in prob simplex; outputs
    predictor's label in prob simplex.
    """
    with tf.device(tf_device), tf.name_scope('bnn_mlp_transformer'):
        tf_vars = []

        x = input_labels
        for i in range(num_layers):
            dense_layer = tf.keras.layers.Dense(
                num_hidden,
                activation=hidden_activation,
                dtype=dtype,
                use_bias=hidden_use_bias,
            )
            x = dense_layer(x)
            for w in dense_layer.weights:
                tf_vars.append(w)

        # output = activation(dot(input, kernel) + bias)
        dense_layer = tf.keras.layers.Dense(
            input_labels.shape[1],
            activation=output_activation,
            use_bias=False,
            dtype=dtype,
            name='bnn_output_pred',
        )
        bnn_out = dense_layer(x)
        for w in dense_layer.weights:
            tf_vars.append(w)

    return bnn_out, tf_vars


def bnn_mlp_loss(*weights, **kwargs):
    with tf.control_dependencies(weights):
        # Given the tf.variables, assign the new values to them
        assign_op = []
        for i, w in enumerate(weights):
            assign_op.append(tf.assign(kwargs['tf_vars'][i], w))

        diff_mvn = tfp.distributions.MultivariateNormalDiag(
            tf.zeros(kwargs['bnn_out'].shape[1]),
            scale_identity_multiplier=kwargs['scale_identity_multiplier'],
        )

        with tf.control_dependencies(assign_op):
            diff = kwargs['bnn_out'] - kwargs['tf_labels']
            kwargs['ops']['diff'] = diff

            loss = tf.reduce_sum(
                diff_mvn.log_prob(diff),
            name='log_prob_dist_sum')

            kwargs['ops']['loss'] = loss

            return loss


def bnn_all_loss(*weights, **kwargs):
    # build ANN
    bnn_out, tf_vars = bnn_mlp(**kwargs['bnn_args'])

    # assign weights
    assign_op = []
    for i, w in enumerate(weights):
        assign_op.append(tf.assign(tf_vars[i], w))

    with tf.control_dependencies(assign_op):
        # get diff and log prob.
        diff_mvn = tfp.distributions.MultivariateNormalDiag(
            tf.zeros(bnn_out.shape[1]),
            scale_identity_multiplier=kwargs['scale_identity_multiplier'],
        )
        diff = bnn_out - kwargs['tf_labels']

        return tf.reduce_sum(
            diff_mvn.log_prob(diff),
            name='log_prob_dist_sum',
        )


def get_bnn_transform(
    input_labels,
    output_labels,
    bnn_args=None,
    num_samples=int(1e4),
    burnin=int(1e4),
    lag=int(1e3),
    parallel_iter=16,
    hyperbolic=False,
    kernel_id='RandomWalkMetropolis',
    kernel_args=None,
    scale_identity_multiplier=1.0,
    random_seed=None,
    dtype=tf.float32,
):
    if input_labels.shape != output_labels.shape:
        raise ValueError(
            'input_labels and output_labels must have the same shape.',
        )
    if bnn_args is None:
        bnn_args = {}

    # Data placeholders
    #tf_input = tf.constant(
    tf_input = tf.placeholder(
        dtype=dtype,
        shape=[None, input_labels.shape[1]],
        name='input_label',
    )
    #tf_labels = tf.constant(
    tf_labels = tf.placeholder(
        dtype=dtype,
        shape=[None, output_labels.shape[1]],
        name='output_labels',
    )

    # Create the BNN model
    _, tf_vars_init = bnn_mlp(tf_input, **bnn_args)
    bnn_args['input_labels'] = tf_input

    # Get loss function
    loss_fn = lambda *w: bnn_all_loss(
        *w,
        #tf_vars=tf_vars,
        #bnn_out=bnn_out,
        bnn_args=bnn_args,
        tf_labels=tf_labels,
        scale_identity_multiplier=scale_identity_multiplier,
    )

    # Get the MCMC Kernel
    kernel = get_mcmc_kernel(loss_fn, kernel_id, kernel_args)

    # Fit the BNN with the MCMC kernel
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_samples,
        current_state=tf_vars_init,
        kernel=kernel,
        num_burnin_steps=burnin,
        num_steps_between_results=lag,
        parallel_iterations=parallel_iter,
    )

    results_dict = {
        'samples': samples,
        'trace': trace,
        #'bnn_out': bnn_out,
        #'tf_vars': tf_vars,
    }

    feed_dict = {
        tf_input: input_labels,
        tf_labels: output_labels,
    }

    return results_dict, feed_dict


def bnn_mlp_run_sess(results_dict, feed_dict):
    # TODO run the session.
    with tf.Session() as sess:
        sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

        iter_results = sess.run(results_dict, feed_dict=feed_dict)

    return iter_results


def assign_weights_bnn(
    weights_sets,
    tf_vars,
    bnn_out,
    input_labels,
    tf_input,
    output_labels=None,
    dtype=tf.float32,
):
    """Given BNN weights and tensors with data, forward pass through network."""
    # assign weights to BNN and create a placeholder for each different tensor
    assign_op = []
    tf_vars_placeholders = []
    for i, w in enumerate(weights_sets):
        weights_placeholder = tf.placeholder(dtype, w.shape[1:])
        tf_vars_placeholders.append(weights_placeholder)
        assign_op.append(tf.assign(tf_vars[i], weights_placeholder))

    feed_dict = {tf_input: input_labels}
    results_list = [bnn_out]

    if output_labels:
        tf_output = tf.placeholder(
            dtype=dtype,
            shape=[None, output_labels.shape[1]],
            name='output_labels',
        )
        results_list.append(bnn_out - tf_output)

        feed_dict[tf_output] = output_labels

    with tf.Session() as sess:
        sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

        # Loop through weights and get the outputs
        iter_results = []
        for sample_idx in len(weights_sets[0].shape[0]):
            for i, var_ph in enumerate(tf_vars_placeholders):
                feed_dict[var_ph] = weights_sets[sample_idx, i]

            iter_results.append(sess.run(results_list, feed_dict=feed_dict))

    return iter_results
