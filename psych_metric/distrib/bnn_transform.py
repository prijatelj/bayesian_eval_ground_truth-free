"""The Bayesian Neural Network that learns the transformation of the target
distribution to the predictor distribution, which results in the conditional
distribution of the predictor conditional on the target.
"""

import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distrib.mcmc import get_mcmc_kernel

def bnn_mlp(
    input_labels,
    num_layers=2,
    num_hidden=10,
    activation=tf.math.sigmoid,
    dtype=tf.float32,
):
    """BNN of a simple MLP model. Input: labels in prob simplex; outputs
    predictor's label in prob simplex.
    """
    with tf.name_scope('bnn_mlp_transformer') as scope:
        tf_vars = []

        x = input_labels
        for i in range(num_layers):
            dense_layer = tf.keras.layers.Dense(
                num_hidden,
                activation=tf.math.sigmoid,
                dtype=dtype,
            )
            x = dense_layer(x)
            for w in dense_layer.weights:
                tf_vars.append(w)

        # output = activation(dot(input, kernel) + bias)
        dense_layer = tf.keras.layers.Dense(
            input_labels.shape[1],
            activation=activation,
            use_bias=False,
            dtype=dtype,
            name='bnn_output_pred',
        )
        bnn_out = dense_layer(x)
        for w in dense_layer.weights:
            tf_vars.append(w)

    return bnn_out, tf_vars


def bnn_mlp_keras(
    dims,
    num_layers=2,
    num_hidden=10,
    activation=tf.math.sigmoid,
    dtype=tf.float32,
):
    target = tf.keras.layers.Input(shape=dims)

    x = target
    for i in range(num_layers):
        x = tf.keras.layers.Dense(
            num_hidden,
            activation=activation,
            dtype=dtype,
        )(x)

    # output = activation(dot(input, kernel) + bias)
    bnn_out = tf.keras.layers.Dense(
        dims,
        activation=activation,
        use_bias=False,
        dtype=dtype,
        name='bnn_output_pred',
    )(x)

    return tf.keras.models.Model(inputs=[target], outputs=[bnn_out])


def bnn_mlp_keras_loss(*weights, **kwargs):
    tf.control_dependencies(weights)
    #kwargs['model'].set_weights(weights)
    # given the tf.variables, assign the new values to them
    for i, w in enumerate(weights):
        tf.assign(kwargs['tf_vars'][i], weights[i])

    return tf.convert_to_tensor(
        kwargs['model'].evaluate(kwargs['target'], kwargs['pred']),
        tf.float32,
    )


def bnn_mlp_loss(*weights, **kwargs):
    tf.control_dependencies(weights)
    #kwargs['model'].set_weights(weights)
    # given the tf.variables, assign the new values to them
    for i, w in enumerate(weights):
        tf.assign(kwargs['tf_vars'][i], weights[i])

    return kwargs['loss']


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
    tf_input = tf.placeholder(
        dtype,
        [None, input_labels.shape[1]],
        name='input_label',
    )
    tf_labels = tf.placeholder(
        dtype,
        [None, output_labels.shape[1]],
        name='output_labels',
    )

    # Create the BNN model
    #bnn_model = bnn_mlp_keras(input_labels.shape[1], **bnn_args)
    #bnn_model.compile('adam', 'categorical_crossentropy')
    bnn_out, tf_vars = bnn_mlp(tf_input, **bnn_args)
    loss = tf.reduce_sum(tf.norm(bnn_out - tf_labels), name='l2_loss_sum')

    # Get the tf vars to be updated from the model in order
    #tf_vars = []
    #for l in bnn_model.layers:
    #    for w in l.weights:
    #        tf_vars.append(w)

    # Get loss function
    loss_fn = lambda *w: bnn_mlp_loss(
        *w,
        tf_vars=tf_vars,
        loss=loss,
    )

    # Get the MCMC Kernel
    kernel = get_mcmc_kernel(loss_fn, kernel_id, kernel_args)

    # Fit the BNN with the MCMC kernel
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_samples,
        current_state=tf_vars,
        kernel=kernel,
        num_burnin_steps=burnin,
        num_steps_between_results=lag,
        parallel_iterations=parallel_iter,
    )

    results_dict = {
        'samples': samples,
        'trace': trace,
        'loss': loss,
        'bnn_out': bnn_out,
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
