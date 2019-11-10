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
    scale : np.ndarray(float)
        The scale matrix (shape matrix Sigma) of the multivariate student T
        distribution.
    """

    def __init__(self, df, loc, scale):
        if not isinstance(df, int):
            raise TypeError(f'`df` must be of type integer, not{type(df)}')
        if not isinstance(loc, np.ndarray):
            raise TypeError(f'`loc` must be of type np.ndarray, not {type(loc)}')
        if loc.dtype != np.float:
            raise TypeError(f'`loc` elements must be of type float, not {loc.dtype}')
        if not isinstance(scale, np.ndarray):
            raise TypeError(f'`loc` must be of type np.ndarray, not {type(df)}')
        if scale.dtype != np.float:
            raise TypeError(f'`scale` elements must be of type float, not {scale.dtype}')
        # TODO check if scale is a valid positive definite matrix

        self.df = df
        self.loc = loc
        self.scale = scale

    def log_prob(self, x):
        return self.log_probability(x, self.df, self.loc, self.scale)

    def log_probability(self, x, df, loc, scale):
        dims = len(loc)
        return (
            scipy.special.gammaln((df + dims) / 2)
            - (df + dims) / 2 * (
                1 + (1 / df) * (x - loc) * np.linalg.inv(scale) * (x - loc)
            ) - (
                scipy.special.gammaln(df / 2)
                + .5 * (dims * (np.log(df) + np.log(np.pi))
                    + np.log(np.linalg.norm(scale))
                )
            )
        )

    def mvst_neg_log_prob(self, x, data, constraint_multiplier=1e5):
        """the Log probability density function of multivariate

        mean is kept constant to the datas column means.
        """
        dims = data.shape[0]

        # expand the params into their proper forms
        df = x[0]
        #loc = x[1 : dims + 1]
        loc = data.mean(0)
        #scale = np.reshape(x[dims + 1:], [dims, dims])
        scale = np.reshape(x[1:], [dims, dims])

        scale = (scale + scale.T) / 2

        neg_log_prob = -self.log_probability(data, df, loc, scale).sum()

        param_constraints = 0

        # apply constraints to variables
        if df <= 0:
            param_constraints += (-df + 1e-3) * constraint_multiplier

        # TODO How to add a constraint to the positive definite matrix `scale`?
        # could apply the same constraint to the different diagonal values.

        return neg_log_prob + param_constraints

    def nelder_mead_mvstudent(
        self,
        data,
        df=None,
        loc=None,
        scale=None,
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
        if scale is None:
            scale = np.cov(data)

        init_data = np.concatenate([[df], loc, scale.flatten()])

        opt_result = scipy.optimize.minimize(
            lambda x: self.mvst_neg_log_prob(x, data),
            # TODO init_data,
            args=[data],
            method='Nelder-Mead',
            maxiter=max_iter,
        )

        assert(opt_result.success)

        opt_x = opt_result.x

        # unpackage the parameters
        df = opt_x[0]
        loc = opt_x[1 : data.shape[1] + 1]
        scale = np.reshape(
            opt_x[data.shape[1] + 1:],
            [data.shape[1], data.shape[1]],
        )

        return df, loc, scale
