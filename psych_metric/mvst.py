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

    def __init__(self, df, loc, sigma, scale=False):
        if not isinstance(df, int):
            raise TypeError(f'`df` must be of type integer, not{type(df)}')
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

    def log_prob(self, x):
        return self.log_probability(x, self.df, self.loc, self.sigma)

    def log_probability(self, x, df, loc, sigma):
        dims = len(loc)

        # TODO Ensure this is broadcastable
        return (
            scipy.special.gammaln((df + dims) / 2)
            - (df + dims) / 2 * (
                1 + (1 / df) * (x - loc) * np.linalg.inv(sigma) * (x - loc)
            ) - (
                scipy.special.gammaln(df / 2)
                + .5 * (dims * (np.log(df) + np.log(np.pi))
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
        estimate_loc=False,
        estimate_scale=False,
        constraint_multiplier=1e5,
    ):
        """the Log probability density function of multivariate

        mean is kept constant to the datas column means.
        """
        dims = data.shape[0]

        # expand the params into their proper forms
        if isinstance(x, np.ndarray):
            df = x[0]
        else:
            df = x

        if estimate_loc:
            # Estimating loc means it is part of x and needs restructured
            loc = x[1 : dims + 1]

            if estimate_scale:
                scale = np.reshape(x[dims + 1:], [dims, dims])
                sigma = scale @ scale.T
            else:
                sigma = np.cov(data) * (df - 2) / df
        else:
            # Derive loc as a constant from the data
            loc = data.mean(0)
            if estimate_scale:
                scale = np.reshape(x[1:], [dims, dims])
                sigma = scale @ scale.T
            else:
                sigma = np.cov(data) * (df - 2) / df

        # Get the negative log probability of the data
        loss = - self.log_probability(data, df, loc, sigma).sum()

        # apply constraints to variables
        if df <= 2:
            loss += (-df + 2 + 1e-3) * constraint_multiplier

        if estimate_scale:
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

        return loss

    def nelder_mead_mvstudent(
        self,
        data,
        df=3,
        loc=None,
        sigma=None,
        const=None,
        max_iter=20000,
        nelder_mead_args=None,
        random_seed=None,
        name='nelder_mead_multivarite_student_t',
        constraint_multiplier=1e5,
    ):
        """Estimates the Maximum Likelihood Estimated of the Multivariate Student
        using Nelder-Mead optimization to minimize the negative log-likelihood
        """
        if nelder_mead_args is None:
            optimizer_args = {}
        if random_seed:
            np.random.seed(random_seed)

        if loc is None:
            loc = data.mean(0)
            estimate_loc = False
        else:
            estimate_loc = True

        if sigma is None:
            sigma = np.cov(data) * (df - 2) / df
            estimate_scale = True
        else:
            estimate_scale = True

        if loc is None and sigma is None:
            init_data = df
        else:
            init_data = np.concatenate([[df], loc, sigma.flatten()])

        opt_result = scipy.optimize.minimize(
            lambda x: self.mvst_neg_log_prob(x, data),
            init_data,
            args=[data],
            method='Nelder-Mead',
            options={'maxiter': max_iter},
        )

        assert(opt_result.success)

        opt_x = opt_result.x

        # unpackage the parameters
        if not (estimate_loc or estimate_scale):
            return opt_x[0]

        df = opt_x[0]
        loc = opt_x[1 : data.shape[1] + 1]
        scale = np.reshape(
            opt_x[data.shape[1] + 1:],
            [data.shape[1], data.shape[1]],
        )

        return df, loc, scale
