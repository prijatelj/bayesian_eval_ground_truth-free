"""K Nearest Neighbors Density Estimation."""
from multiprocessing import Pool

import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors

from psych_metric.distrib import distrib_utils


def knnde_log_prob(samples, knn, k, knn_density_num_samples):
    """Empirically estimates the predictor log probability using K Nearest
    Neighbors Density Estimation.
    """
    if len(samples.shape) == 1:
        # Handle single sample
        samples = samples.reshape(1, -1)

    radius = knn.kneighbors(samples, k)[0][:, -1]

    # log(k) - log(n) - log(volume)
    log_prob = np.log(k) - np.log(knn_density_num_samples)

    volume = samples.shape[1] * (np.log(np.pi) / 2 + np.log(radius)) \
            - scipy.special.gammaln(samples.shape[1] / 2 + 1)

    # calculate the n sphere volume being contained w/in the n simplex
    return log_prob - volume

# TODO change to euclid_differences instead! NOT used for BNN
def euclid_diff_knn_log_prob_single(
    target,
    pred,
    simplex_transform,
    transform_knn_diffs,
    n_neighbors=10, # 1 is lowest bias
    n_jobs=1,
):
    """Calculate the log prob of a single sample of a conditional probability
    using a Euclidean simplex space conditional prob model and KNN to estimate
    the density.
    """
    # Find valid differences from saved set: `self.transform_knn_diffs`
    # to test validity, convert target sample to simplex space.
    simplex_target = simplex_transform.to(target)

    # add differences to target & convert back to full dimensional space.
    diff_check = simplex_transform.back(transform_knn_diffs + simplex_target)

    # Check which are valid samples. Save indices or new array
    valid_diffs = transform_knn_diffs[
        np.where(distrib_utils.is_prob_distrib(diff_check))[0]
    ]

    # TODO should check if number of valid_diffs >> k, otherwise not accurate estimate.

    # Fit KNN to the differences valid to the specific target.
    knn = NearestNeighbors(n_neighbors, n_jobs=n_jobs)
    knn.fit(valid_diffs)

    # Get distance between actual sample pair of target and pred
    actual_diff = simplex_transform.to(pred) - simplex_target

    # Estimate the log probability.
    return knnde_log_prob(actual_diff, knn, n_neighbors, len(valid_diffs))


def euclid_diff_knn_log_prob(
    given,
    pred,
    simplex_transform,
    transform_knn_diffs,
    n_neighbors=10,
    n_jobs=1,
):
    with Pool(processes=n_jobs) as pool:
        log_prob = pool.starmap(
            euclid_diff_knn_log_prob_single,
            zip(
                given,
                pred,
                [simplex_transform] * len(given),
                [transform_knn_diffs] * len(given),
                [n_neighbors] * len(given),
            ),
        )

    return np.array(log_prob)


def euclid_bnn_knn_log_prob_single(
    target,
    pred,
    simplex_transform,
    bnn,
    weight_sets,
    n_neighbors=10, # 1 is lowest bias
    n_jobs=1,
):
    """Calculate the log prob of a single sample of a conditional probability
    using a Euclidean simplex space conditional prob modeled via BNN and KNN to
    estimate the density.
    """
    simplex_target = simplex_transform.to(target)

    bnn_output = bnn.predict(simplex_target, weight_sets)

    output_check = simplex_transform.back(bnn_output)

    # Check which are valid samples. Save indices or new array
    valid_diffs = bnn_output[
        np.where(distrib_utils.is_prob_distrib(output_check))[0]
    ]

    # TODO should check if number of valid_diffs >> k, otherwise not accurate estimate.

    # Fit KNN to the differences valid to the specific target.
    knn = NearestNeighbors(n_neighbors, n_jobs=n_jobs)
    knn.fit(valid_diffs)

    # Estimate the log probability.
    return knnde_log_prob(
        simplex_transform.to(pred),
        knn,
        n_neighbors,
        len(valid_diffs),
    )


def euclid_bnn_knn_log_prob(
    given,
    pred,
    simplex_transform,
    bnn,
    weight_sets,
    n_neighbors=10,
    n_jobs=1,
):
    with Pool(processes=n_jobs) as pool:
        log_prob = pool.starmap(
            euclid_bnn_knn_log_prob_single,
            zip(
                given,
                pred,
                [simplex_transform] * len(given),
                [bnn] * len(given),
                [weight_sets] * len(given),
                [n_neighbors] * len(given),
            ),
        )

    return np.array(log_prob)
