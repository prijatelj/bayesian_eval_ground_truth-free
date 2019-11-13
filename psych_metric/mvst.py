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
        if not isinstance(samples, np.ndarray):
            try:
                samples = np.array(samples)
            except:
                raise(f'`samples` is expected to convertible to a numpy array.')

        if len(samples.shape) == 2 and samples.shape[1] == len(loc):
            # Handle multiple samples
            return np.array([
                self.log_probability(x, self.df, self.loc, self.sigma)
                for x in samples
            ])
        elif len(samples.shape) == 1 and len(samples) == len(loc):
            # Handle the single sample case
            return self.log_probability(x, self.df, self.loc, self.sigma)
        else:
            raise TypeError(' '.join([
                'Expected given `samples` to be of either', 'dimension 2 when',
                'multiple samples (with the rows as samples) or 1 when a',
                'single sample. The samples must have the same dimensions as',
                '`loc`.',
            ]))

    def _log_probability(self, sample, df, loc, sigma):
        """Returns the log probability of a single sample."""
        dims = len(loc)
        return (
            scipy.special.gammaln((df + dims) / 2)
            - (df + dims) / 2 * (
                1 + (1 / df) * (sample - loc) @ np.linalg.inv(sigma)
                @ (sample - loc).T
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
        const=None,
    ):
        """the Log probability density function of multivariate

        mean is kept constant to the datas column means.
        """
        dims = data.shape[1]

        # expand the params into their proper forms
        if isinstance(const, dict) and 'df' in const:
            df = const['df']
            loc = x[: dims]
            scale = np.reshape(x[dims:], [dims, dims])
        else:
            df = x[0]
            loc = x[1 : dims + 1]
            scale = np.reshape(x[dims + 1:], [dims, dims])

        # Get Sigma Matrix from scale.
        sigma = scale @ scale.T

        # Get the negative log probability of the data
        loss = - self.log_prob(data, df, loc, sigma).sum()

        # apply constraints to variables
        if not (isinstance(const, dict) and 'df' in const) and df <= 0:
            loss += (1e-4 - df) * constraint_multiplier

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
        const=None,
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

        if const and 'df' in const:
            print(f'df is const. df = {self.df}.')

            init_data = np.concatenate([self.loc, self.sigma.flatten()])
            return scipy.optimize.minimize(
                lambda x: self.mvst_neg_log_prob(x, data, const={'df': self.df}),
                init_data,
                method='Nelder-Mead',
                options={'maxiter': max_iter},
            )
        else:
            init_data = np.concatenate([[self.df], self.loc, self.sigma.flatten()])
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
