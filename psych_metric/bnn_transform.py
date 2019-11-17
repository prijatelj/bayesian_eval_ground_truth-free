"""The Bayesian Neural Network that learns the transformation of the target
distribution to the predictor distribution, which results in the conditional
distribution of the predictor conditional on the target.
"""

import tensorflow as tf
import tensorflow as tfp

def bnn_mlp_inference(
    dims,
    num_samples=1e4,
    burnin=1e4,
    lag=1e3,
    num_layers=2,
    num_hidden=10,
    activation='sigmoid',
    dtype=tf.float32,
    parallel_iter=16,
    hyperbolic=False,
    random_seed=None,
):
    """BNN of a simple MLP model. Input: labels in prob simplex; outputs
    predictor's label in prob simplex.
    """
    transform_session = tf.Session()

    with tf.name_scope('bnn_mlp_transformer') as scope:
        target = tf.placeholder(dtype, [dims], name='target_label')
        actual_pred = tf.placeholder(dtype, [dims], name='target_label')

        x = target
        for i in range(num_layers):
            x = tf.keras.layers.Dense(
                num_hidden,
                activation=tf.math.sigmoid,
                dtype=dtype,
            )(x)

        # output = activation(dot(input, kernel) + bias)
        pred = tf.keras.layers.Dense(
            dims,
            activation=tf.math.sigmoid,
            use_bias=False,
            dtype=dtype,
            name='pred',
        )(x)

        loss = tf.nn.l2_loss(pred - actual_pred)

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

        # Train it using RandomWalk or NoUTurnSampler
        samples, _ = tfp.mcmc.sample_chain(
            num_results=num_samples,
            current_state=pred,
            kernel=kernel,
            num_burnin_steps=burnin,
            num_steps_between_results=lag,
            parallel_iterations=parallel_iter,
        )

