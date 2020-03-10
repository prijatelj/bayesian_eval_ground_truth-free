"""General measure funcitons for ease of executing an arbitrary measure on data
that may hav have multiple values per sample, such as having an empirical
sampling of a conditional distribution in order to obtain a distribution of the
measure.

Notes
-----
All this assumes discrete distributions.
"""
import numpy as np

# TODO consider adding axes for: conditionals_axis=None, target_axis=0
def measure(measure_func, targets, preds):
    """

    Parameters
    ----------
    measure_func : callable
        Function that performs some measure on the target and pred expecting
        them in order of targets and preds (ie. measure_func(targets, preds).
    targets : np.ndarray
        targets is a numpy array with shape of [number of targets, classes],
        where classes is the number of classes or discrete dimensions of a
        sample.
    preds : np.ndarray
        preds is a numpy array with shape of [number of targets, classes],
        where classes is the number of classes or discrete dimensions of a
        sample, XOR preds has the shape of [number of targets, conditionals,
        classes] where conditionals is the number of multiple pred samples per
        target sample. This latter shape is
    """
    if len(preds.shape) in {1, 2}:
        # Assumed structure is then shape of 2, and non-target axis is the
        # n-dim discrete sample
        return measure_func(targets, preds)

    if len(preds.shape) != 3 or len(targets.shape) != 2:
        raise ValueError(' '.join([
            'preds and targets are expected to have 3 and 2 dimensions',
            'respectively.',
        ]))

    # NOTE assumes preds of shape [targets, conditionals, classes]

    conditionals_measurements = []
    for c_idx in range(preds.shape[1]):
        conditionals_measurements.append(
            measure_func(targets, preds[:, c_idx, :]),
        )

    return np.array(conditionals_measurements)


def highest_density_credible_interval(vector, sample_density):
    """Finds the highest density credible interval (aka highest posterior
    density interval) for a vector of values. This interval is found by sorting
    the vector vlaues, then iterating through the data checking for which
    interval of some % size (ie. 95% credible interval) has the smallest
    difference of high quantile - low quantile. Given a static interval size,
    the smallest distance in values of the given vector is the highest density
    credible interval of that size.

    The granularity used to check the intervals is always by 1 sample.

    Parameters
    ----------
    vector : np.ndarray
        The vector of values whose credible interval is to be calculated.
    sample_density : float
        Float from (0,1) that represents the amount of data points that the
        credible interval is to contain (ie. .95 results in a credible interval
        that contains 95% of the sorted data with the highest density).

    Returns
    -------
    (low_quantile, high_quantile) : tuple
        A tuple of the lower and upper quantiles that bound the highest density
        crediblity interval of the given size.
    """
    if sample_density <= 0 or sample_density >= 1:
        raise ValueError(
            'The Credible interval size must be within the range (0,1)',
        )

    if not isinstance(vector, np.ndarray)
        vector = np.array(vector)

    if vector.shape > 1:
        # If given multiple dim array, flatten into 1d
        vector = np.ravel(vector)

    # Sort vector
    sorted_vec = np.sort(vector)
    vec_size = len(sorted_vec)

    # Get the paired low and high quantiles
    high_quantiles = sorted_vec[int(np.ceil(vec_size * sample_density)):]
    low_quantiles = sorted_vec[:int(np.floor(vec_size * (1.0 - sample_density)))]

    # Iterate through the vector with static interval size, check differences
    # minimum distance is the highest density credible interval
    idx = np.argmin(high_quantiles - low_quantiles)

    return low_quantiles[idx], high_quantiles[idx]
