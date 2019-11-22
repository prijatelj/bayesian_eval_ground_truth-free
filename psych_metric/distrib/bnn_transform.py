"""The Bayesian Neural Network that learns the transformation of the target
distribution to the predictor distribution, which results in the conditional
distribution of the predictor conditional on the target.
"""

import tensorflow as tf
import tensorflow_probability as tfp


def bnn_mlp_inference(
    dims,
    num_layers=2,
    num_hidden=10,
    activation=tf.math.sigmoid,
    dtype=tf.float32,
):
    """BNN of a simple MLP model. Input: labels in prob simplex; outputs
    predictor's label in prob simplex.
    """
    transform_session = tf.Session()

    with tf.name_scope('bnn_mlp_transformer') as scope:
        target = tf.placeholder(dtype, [dims], name='target_label')
        pred = tf.placeholder(dtype, [dims], name='pred_label')

        x = target
        for i in range(num_layers):
            x = tf.keras.layers.Dense(
                num_hidden,
                activation=tf.math.sigmoid,
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

    return bnn_out, target, pred


def foo(
    bnn_args,
    num_samples=1e4,
    burnin=1e4,
    lag=1e3,
    parallel_iter=16,
    hyperbolic=False,
    random_seed=None,
):
    # Create the BNN model
    bnn_out, target, pred = bnn_mlp_inference(**bnn_args)

    # Get the loss
    loss = tf.nn.l2_loss(bnn_out - pred)

    # Get the MCMC Kernel
    if hyperbolic:
        raise NotImplementedError

        kernel = tfp.mcmc.NoUTurnSampler(
            loss,
            step_size,
            seed=random_seed,
        )
    else:
        kernel = tfp.mcmc.RandomWalkMetropolis(
            loss,
            seed=random_seed,
        )

    # Fit the BNN with the MCMC kernel
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_samples,
        current_state=bnn_out,
        kernel=kernel,
        num_burnin_steps=burnin,
        num_steps_between_results=lag,
        parallel_iterations=parallel_iter,
    )
