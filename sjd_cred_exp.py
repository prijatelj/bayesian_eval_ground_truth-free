"""The Experiment with SJD to show how its predictions and metrics with
Credible Intervals compare to existing methods (which all lack credible
intervals).
"""

def sjd_metric_cred():
    """Get a single model trained on all data, and create many samples to
    obtain the credible interval of the predictor output, joint distribution,
    and metrics calculated on that joint distribution.
    """
    # Fit SJD to human & preds if not already available.

    # sample many times from that SJD (1e6)

    # optionally save those samples to file.

    # Calculate credible interval given alpha for Joint Distrib samples

    # compare how that corresponds to the actual data human & pred.

    # save all this and visualize it.

    # Also, able to quantify % of data points outside of credible interval

    raise NotImplementedError()


