"""The functions for converting to and from the Hyperbolic simplex basis"""
import logging

import numpy as np


def cartesian_to_hypersphere(vectors):
    """Convert from Cartesian coordinates to hyperspherical coordinates of the
    same n-dimensions.

    Attributes
    ----------
    vectors : np.ndarray
        2-dimensional array where the first dimension is the samples and the
        second is the n-dimensional Cartesian coordinates.

    Results
    -------
    np.ndarray
        2-dimensional array where the first dimension is the samples and the
        second dimension contains n elements consisting of the n-1-dimensional
        hyperspherical coordinates where the first element is the radius, and
        then followed by n-1 angles for each dimension.
    """
    radii = np.linalg.norm(vectors[:, 0])


    return


def hypersphere_to_cartesian(vectors):
    """Convert from hyperspherical coordinates to Cartesian coordinates of the
    same n-dimensions.

    Attributes
    -------
    vectors : np.ndarray
        2-dimensional array where the first dimension is the samples and the
        second dimension contains n elements consisting of the n-1-dimensional
        hyperspherical coordinates where the first element is the radius, and
        then followed by n-1 angles for each dimension.

    Results
    -------
    np.ndarray
        2-dimensional array where the first dimension is the samples and the
        second is the n-dimensional Cartesian coordinates.
    """
    if len(vectors.shape) == 1:
        # single sample
        vectors = vectors.reshape(1, -1)

    # x1 = radius * cos(rho_1)
    # xn-1 = radius * sin(rho_1) * ... * sin(rho_n-2) * cos(rho_n-1)
    # xn = radius * sin(rho_1) * ... * sin(rho_n-1)
    sin = np.concatenate(
        (
            np.ones([vectors.shape[0], 1]),
            np.cumprod(np.sin(vectors[:, 1:]), axis=1)
        ),
        axis=1,
    )
    cos = np.concatenate(
        (np.cos(vectors[:, 1:]), np.ones([vectors.shape[0], 1])),
        axis=1,
    )
    return  vectors[:, 0].reshape(-1, 1) * sin * cos


class HyperbolicSimplexTransform(object):
    """

    Attributes
    ----------
    origin_adjust : np.ndarray

    """

    def __init__(self, dim):
        # Create origin adjustment, center of simplex at origin
        self.origin_adjust = np.ones(dim) / 2

        raise NotImplementedError

    def to(self, vectors):
        """Transform given vectors into hyperbolic probability simplex space."""
        # Center discrete probability simplex at origin in Euclidean space
        centered_vectors = vectors - self.origin_adjust

        # Convert to polar coordinates

        # Stretch simplex into hypersphere, no longer conserving the angles

        # Inverse Poincare' Ball method to go into hyperbolic space

        return

    def tf_to(self, vectors):
        """Transform given vectors into n-1 probability simplex space done in
        tensorflow code.

        """
        # center 1st dim's extreme value at origin
        return

    def back(self, vectors):
        """Transform given vectors out of n-1 probability simplex space."""
        return

    def tf_from(self, vectors):
        """Transform given vectors out of n-1 probability simplex space done in
        tensorflow code.
        """
        return
