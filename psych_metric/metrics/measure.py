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

    if not (len(preds.shape) != 3 and len(targets.shape) != 2):
        raise ValueError(' '.join([
            'preds and targets are expected to have 3 and 2 dimensions',
            'respectively.',
        ]))

    # NOTE assumes preds of shape [targets, conditionals, classes]

    conditionals_measurements = []
    for c_idx in range(len(preds.shape[1])):
        conditionals_measurements.append(
            measure_func(targets, preds[:, c_idx, :]),
        )

    return np.array(conditionals_measurements)
