"""The Class and related functions for a BNN implemented via MCMC."""
from copy import deepcopy

import tensorflow as tf

from psych_metric.distrib import bnn_transform


class BNNMCMC(object):
    """An implementation of a Bayesian Neural Network implemented via Markov
    Chain Monte Carlo.

    Attributes
    ----------
    hidden_activation : function
        Defaults to tf.math.sigmoid. None is same as linear.
    hidden_use_bias : bool
    output_activation : function
        Default to None, thus linear. example is tf.math.sigmoid
    output_use_bias : bool
    kernel :
    sess : tf.Session
    """

    def __init__(
        self,
        dim,
        num_layers=1,
        num_hidden=10,
        hidden_activation=tf.math.sigmoid,
        hidden_use_bias=True,
        output_activation=None, #, tf.math.sigmoid,
        output_use_bias=True, # False,
        #kernel_args, # TODO
        dtype=tf.float32,
        #sess=None,
        sess_config=None,
    ):
        self.dtype = dtype
        self.sess_config = sess_config

        # Make BNN and weight_placeholders, input & output tensors
        self.input = tf.placeholder(
            dtype=dtype,
            shape=[None, dim],
            name='input_labels',
        )

        self.output, self.weight_placeholders = bnn_transform.bnn_mlp_placeholders(
            self.input,
            num_layers,
            num_hidden,
            hidden_activation,
            hidden_use_bias,
            output_activation,
            output_use_bias,
            dtype,
        )

        # TODO need to add actual fitting via MCMC, sampling, etc.
        # for now, just a way to better contain the args and functions.

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo):
        raise NotImplementedError

        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        for k, v in self.__dict__.items():
            setattr(new, k, deepcopy(v, memo))

        return new

    def fit(self, kernel_args):

        raise NotImplementedError(' '.join([
            'The API does not yet contain the BNN MCMC training code. It',
            'exists in prototype format as a series of functions.',
            'See `bnn_exp.py`',
        ]))

    def get_weight_sets(self, num):
        """After fitting the BNN via MCMC, get a set of weights for different
        instances of the BNN to obtian the distribution of outputs.
        """
        raise NotImplementedError(' '.join([
            'The API does not yet contain the BNN MCMC sampling code. It',
            'exists in prototype format as a series of functions.',
            'See `bnn_exp.py`',
        ]))

    def predict(self, given_samples, weight_sets, sess_config=None):
        """Returns the predictions of the BNN with the given weight_sets.

        Returns
        -------
        np.ndarray
            A list or an array of shape (num_given_samples,
            num_valid_weight_sets_per_given_sample, discrete distrib
            dimensions). In the case of Euclidean transform, the returned
            array may contain some invalid discrete probability distributions.
        """
        if sess_config is None and self.sess_config is not None:
            sess_config = self.sess_config

        return bnn_transform.assign_weights_bnn(
            weight_sets,
            self.weight_placeholders,
            self.output,
            given_samples,
            self.input,
            dtype=self.dtype,
            sess_config=sess_config, # TODO, replace with class attrib?
        )
