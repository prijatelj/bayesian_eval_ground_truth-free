"""K Nearest Neighbors Density Estimation."""
from multiprocessing import Pool

import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors

from psych_metric.distrib import distrib_utils


def euclid_knn_log_prob(samples, knn, k, knn_density_num_samples):
    """Empirically estimates the predictor log probability using K Nearest
    Neighbors.
    """
    if len(samples.shape) == 1:
        # Handle single sample
        samples = samples.reshape(1, -1)

    radius = knn.kneighbors(samples, k)[0][:, -1]

    # log(k) - log(n) - log(volume)
    log_prob = np.log(k) - np.log(knn_density_num_samples)

    # calculate the n sphere volume being contained w/in the n simplex
    return log_prob - (samples.shape[1] * (np.log(np.pi) / 2 + np.log(radius))
        - scipy.special.gammaln(samples.shape[1] / 2 + 1))


def euclid_transform_knn_log_prob_single(
    trgt,
    pred,
    simplex_transform,
    transform_knn_dists,
    n_neighbors=1, # 1 is lowest bias
    n_jobs=1,
):
    # Find valid differences from saved set: `self.transform_knn_dists`
    # to test validity, convert target sample to simplex space.
    simplex_trgt = simplex_transform.to(trgt)

    # add differences to target & convert back to full dimensional space.
    dist_check = simplex_transform.back(transform_knn_dists + simplex_trgt)

    # Check which are valid samples. Save indices or new array
    valid_dists = transform_knn_dists[
        np.where(distrib_utils.is_prob_distrib(dist_check))[0]
    ]

    # TODO should check if number of valid_dists >> k, otherwise not accurate estimate.

    # Fit KNN to the differences valid to the specific target.
    knn = NearestNeighbors(n_neighbors, n_jobs=n_jobs)
    knn.fit(valid_dists)

    # Get distance between actual sample pair of target and pred
    actual_dist = simplex_transform.to(pred) - simplex_trgt

    # Estimate the log probability.
    return euclid_knn_log_prob(actual_dist, knn, n_neighbors, len(valid_dists))


def euclid_transform_knn_log_prob(
    given,
    pred,
    simplex_transform,
    transform_knn_dists,
    n_neighbors=1,
    n_jobs=1,
):
    with Pool(processes=n_jobs) as pool:
        log_prob = pool.starmap(
            euclid_transform_knn_log_prob_single,
            zip(
                given,
                pred,
                [simplex_transform] * len(given),
                [transform_knn_dists] * len(given),
                [n_neighbors] * len(given),
            ),
        )

    return np.array(log_prob)
