"""The Class and related functions for a BNN implemented via MCMC."""

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
        num_layers=1,
        num_hidden=10,
        hidden_activation=tf.math.sigmoid,
        hidden_use_bias=True,
        output_activation=None, #, tf.math.sigmoid,
        output_use_bias=True, # False,
        kernel_args,
        dtype=tf.float32,
        sess=None,
    ):
        pass

    def fit(self, kernel_args):

        raise NotImplemented(' '.join([
            'The API does not yet contain the BNN MCMC training code. It',
            'exists in prototype format as a series of functions.',
            'See `bnn_exp.py`',
        ])

    def get_weight_sets(self, num):
        """After fitting the BNN via MCMC, get a set of weights for different
        instances of the BNN to obtian the distribution of outputs.
        """
        raise NotImplemented(' '.join([
            'The API does not yet contain the BNN MCMC sampling code. It',
            'exists in prototype format as a series of functions.',
            'See `bnn_exp.py`',
        ])

    def predict(self, given_samples, weight_sets):
        """Returns the predictions of the BNN with the given weight_sets.

        Returns
        -------
        np.ndarray
            A list or an array of shape (num_given_samples,
            num_valid_weight_sets_per_given_sample, discrete distrib
            dimensions). In the case of Euclidean transform, the returned
            array may contain some invalid discrete probability distributions.
        """

        return pred
