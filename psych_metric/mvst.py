"""Multivariate Student T distribution."""
import numpy as np
import scipy


class MultivariateStudentT(object):
    """The Multivariate Student T distribution.

    Attributes
    ----------
    df : int
        The degrees of freedom
    loc : np.ndarray(float)
        The means of the different dimensions experessed as a vector of postive
        floats.
    sigma : np.ndarray(float)
        The sigma matrix of the multivariate student T distribution. This is
        NOT the scale matrix! This is equal to scale @ scale.T
    """

    def __init__(self, df, loc, sigma, scale=True):
        if not isinstance(df, float):
            raise TypeError(f'`df` must be of type float, not{type(df)}')
        if not isinstance(loc, np.ndarray):
            raise TypeError(f'`loc` must be of type np.ndarray, not {type(loc)}')
        if loc.dtype != np.float:
            raise TypeError(f'`loc` elements must be of type float, not {loc.dtype}')
        if not isinstance(sigma, np.ndarray):
            raise TypeError(f'`sigma` must be of type np.ndarray, not {type(df)}')
        if sigma.dtype != np.float:
            raise TypeError(f'`sigma` elements must be of type float, not {sigma.dtype}')
        # TODO check if scale is a valid positive definite matrix

        self.df = df
        self.loc = loc
        self.sigma = sigma @ sigma.T if scale else sigma

    def fit(self, data, method='nelder_mead', *args, **kwargs):
        if method == 'nelder_mead':
            return self.nelder_mead(data, *args, **kwargs)

        raise ValueError(f'Only the Nedler-Mead method is supported, not `{method}`.')

    def log_prob(self, x):
        return self.log_probability(x, self.df, self.loc, self.sigma)

    def log_probability(self, x, df, loc, sigma):
        dims = len(loc)

        # TODO Ensure this is broadcastable
        return (
            scipy.special.gammaln((df + dims) / 2)
            - (df + dims) / 2 * (
                1 + (1 / df) * (x - loc) @ np.linalg.inv(sigma) @ (x - loc).T
            ) - (
                scipy.special.gammaln(df / 2)
                + .5 * (
                    dims * (np.log(df) + np.log(np.pi))
                    + np.log(np.linalg.norm(sigma))
                )
            )
        )

    def sample(self, number_samples):
        """Sample from the distribution."""
        raise NotImplementedError(
            'Need to implement sampling from Multivariate Student T distrib.'
        )

        # TODO sampling, perhaps via MCMC...?
        return samples

    def mvst_neg_log_prob(
        self,
        x,
        data,
        constraint_multiplier=1e5,
    ):
        """the Log probability density function of multivariate

        mean is kept constant to the datas column means.
        """
        dims = data.shape[1]

        # expand the params into their proper forms
        df = x[0]
        loc = x[1 : dims + 1]
        #scale = np.reshape(x[dims + 1: 2 * dims + 1], [dims, dims])
        scale = np.reshape(x[dims + 1:], [dims, dims])
        sigma = scale @ scale.T

        # Get the negative log probability of the data
        loss = - self.log_probability(data, df, loc, sigma).sum()

        # apply constraints to variables
        if df <= 0:
            loss += (-df + 1e-4) * constraint_multiplier

        """
        # Check if scale is a valid positive definite matrix
        try:
            np.linalg.cholesky(scale)
        except:
            # TODO How to add a constraint to the positive definite matrix `scale`?
            # could apply the same constraint to the different diagonal values.

            # it is a boolean state afaik, so if not positive definite, then return
            # high value for loss

            # TODO ensure the use of absolute value is fine.
            loss += loss**2 * constraint_multiplier
        """

        return loss

    def nelder_mead(
        self,
        data,
        #const=None,
        max_iter=10000,
        nelder_mead_args=None,
        name='nelder_mead_multivarite_student_t',
        constraint_multiplier=1e5,
    ):
        """Estimates the Maximum Likelihood Estimated of the Multivariate Student
        using Nelder-Mead optimization to minimize the negative log-likelihood
        """
        if nelder_mead_args is None:
            optimizer_args = {}

        init_data = np.concatenate([[self.df], self.loc, self.sigma.flatten()])

        #opt_result = scipy.optimize.minimize(
        return scipy.optimize.minimize(
            lambda x: self.mvst_neg_log_prob(x, data),
            init_data,
            method='Nelder-Mead',
            options={'maxiter': max_iter},
        )

        #assert(opt_result.success)
        opt_x = opt_result.x

        self.df = opt_x[0]
        self.loc = opt_x[1 : data.shape[1] + 1]
        self.scale = np.reshape(
            opt_x[data.shape[1] + 1:],
            [data.shape[1], data.shape[1]],
        )

        return df, loc, scale
