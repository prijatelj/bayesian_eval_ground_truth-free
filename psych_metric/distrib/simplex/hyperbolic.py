"""The functions for converting to and from the Hyperbolic simplex basis"""
import logging

import numpy as np

from psych_metric.distrib.simplex.euclidean import EuclideanSimplexTransform


def cart2polar(vectors):
    """Convert from 2d Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    vectors : np.ndarray
        2-dimensional array where the first dimension is the samples and the
        second is the 2-dimensional Cartesian coordinates.

    Returns
    -------
        2-dimensional array where the first dimension is the samples and the
        second dimension contains 2 elements consisting of the polar
        coordinates where the first element is the radius, and then followed by
        the angle.
    """
    return np.concatenate(
        (
            np.linalg.norm(vectors, axis=1, keepdims=True),
            np.arctan2(vectors[:, 1], vectors[:, 0]).reshape([-1, 1]),
        ),
        axis=1,
    )


def polar2cart(vectors):
    """Convert from polar to 2d Cartesian coordinates.

    Parameters
    ----------
    vectors : np.ndarray
        2-dimensional array where the first dimension is the samples and the
        second dimension contains 2 elements consisting of the polar
        coordinates where the first element is the radius, and then followed by
        the angle.

    Results
    -------
    np.ndarray
        2-dimensional array where the first dimension is the samples and the
        second is the 2-dimensional Cartesian coordinates.
    """
    return vectors[:, [0]] * np.concatenate(
        (
            np.cos(vectors[:, [1]]),
            np.sin(vectors[:, [1]]),
        ),
        axis=1,
    )


def cartesian_to_hypersphere(vectors):
    """Convert from Cartesian coordinates to hyperspherical coordinates of the
    same n-dimensions.

    Parameters
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
    if len(vectors.shape) == 1:
        # single sample
        vectors = vectors.reshape([1, -1])

    if vectors.shape[1] == 2:
        return cart2polar(vectors)
    elif vectors.shape[1] < 2:
        raise ValueError(' '.join([
            'Expected the number of coordinate dimensions to be >= 2, but',
            f'recieved vectors with shape {vectors.shape}. and axis being',
            f'1.',
        ]))

    #radii = np.linalg.norm(vectors[:, 0])
    flipped = np.fliplr(vectors)
    cumsqrt = np.sqrt(np.cumsum(flipped ** 2, axis=1))
    #radii = cumsqrt[:, -1]
    angles = np.arccos(flipped / cumsqrt)
    # angles 1 -- n-2 = np.fliplr(angles[2:])

    last_angle = np.pi - 2 * np.arctan(
        (flipped[:, 1] + cumsqrt[:, 1]) / flipped[:, 0]
    )

    # radius followed by ascending n-1 angles per row
    return np.concatenate(
        (
            cumsqrt[:, [-1]],
            np.fliplr(angles[:, 2:]),
            last_angle.reshape([-1, 1]),
        ),
        axis=1,
    )


def hypersphere_to_cartesian(vectors):
    """Convert from hyperspherical coordinates to Cartesian coordinates of the
    same n-dimensions.

    Parameters
    ----------
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
        vectors = vectors.reshape([1, -1])

    if vectors.shape[1] == 2:
        return polar2cart(vectors)
    elif vectors.shape[1] < 2:
        raise ValueError(' '.join([
            'Expected the number of coordinate dimensions to be >= 2, but',
            f'recieved vectors with shape {vectors.shape}. and axis being',
            f'1.',
        ]))

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


def get_rotation_axis_angle(dim, dim_to_zero=0):
    """Given number of dimensions of a discrete probability simplex, find some
    vector that lies on the rotation axis and angle to zero the given
    dimension.
    """

    return


def givens_rotation(dim, x, y, angle):
    """Creates a Givens rotation matrix."""
    rotate = np.eye(dim)
    rotate[x, x] = np.cos(angle)
    rotate[x, y] = -np.sin(angle)
    rotate[y, x] = np.sin(angle)
    rotate[y, y] = np.cos(angle)
    return rotate


def rotate_around(rotation_simplex, angle):
    """General n-dimension rotation."""
    if rotation_simplex.shape[0] > rotation_simplex.shape[0]:
        # expects points to contained in the rows, elements in columns
        rotation_simplex = rotation_simplex.T

    # Transpose the simplex st the first point is centered at origin
    v = rotation_simplex - rotation_simplex[0]
    # TODO how to include this in the ending matrix? need to be homogenous
    # coordinates? As is right now, the alg will treat v1 as v0 post-transpose

    n = len(rotation_simplex)
    mat = np.eye(n)

    for r in range(1, n):
        for c in list(range(r, n))[::-1]:
            v = v @ mat
            mat = mat @ givens_rotation(
                n,
                c,
                c - 1,
                np.arctan2(v[r, c], v[r, c-1]),
            )

    return mat @ rotation_matrix(n, n-1, n, angle) @ np.linalg.inv(mat)


def get_simplex_boundary_radius(angles, circumscribed_radius):
    """Returns the radius of the point on the boundary of the regular simplex

    Parameters
    ----------
    angles : np.ndarray
        Array of different sets of angles defining the location of the boundary
        point on the simplex whose radius is being calculated. Each row is
        a different point and the columns are the angles.
    circumscribed_radius :
        The radius of the circumscribed hypersphere of the simplex, which is
        equal to the radius of each vertex of the regular simplex.
    """
    #return circumscribed_radius * np.cos(np.pi / 3) \
    #    / np.cos(2 / 3 * np.pi - angles)




class HyperbolicSimplexTransform(object):
    """

    Attributes
    ----------
    origin_adjust : np.ndarray

    """

    def __init__(self, dim):
        # Create origin adjustment, center of simplex at origin
        #self.origin_adjust = np.ones(dim) / 2
        #self.euclid_simplex_transform = EuclideanSimplexTransform(dim)
        #extremes = np.eye(self.input_dim)
        #simplex_extremes = self.euclid_simplex_transform.to(extremes)

        # TODO equal scaling of dimensions?
        # extremes / 2 is the radii to each vertex of the simplex, and their
        # norms is the radii which are all the same via orthogonal rotation
        # matrix.
        # Save the vector for translating all pts to center about origin.
        #self.centroid = simplex_extremes.mean(axis=0)

        # Center Simplex in N-dim
        self.center_adjust = 1.0 / dim

        # Rotate to zero out one arbitrary dimension, drop that zeroed dim.
        self.rotate_drop_dim = get_rotate_drop_dim_matrix(dim, 0)

        #self.circumscribed_radius = np.linalg.norm()

        # rotate and drop dim
        # does not have to be negative, but be aware of rotation direction
        #self.axis_rot = -np.ones(dim)
        #self.axis_rot[0] = 0

        self.angle_rot = np.arctan2(
            1.0 - 1.0 / dim,
            np.linalg.norm([1.0 / dim] * (dim - 1)),
        )
        # TODO Rotation matrix or Rotors rotate and drop dim

        raise NotImplementedError

    @property
    def input_dim(self):
        return self.euclid_simplex_transform.input_dim

    @property
    def output_dim(self):
        return self.euclid_simplex_transform.output_dim

    def to(self, vectors):
        """Transform given vectors into hyperbolic probability simplex space."""
        # Center discrete probability simplex at origin in Euclidean space
        #centered_vectors = vectors - self.origin_adjust
        # TODO change euclid_simplex / make alt. to use center and rotate only
        #euclid_simplex = self.euclid_simplex_transform.to(vectors)


        # center at origin.
        # Center the euclid n-1 basis of simplex at origin, then check if
        # vertices equidistant, which they should be. then can use those as
        # circumscribed_radius
        centered = vectors - self.centroid



        # Convert to polar/hyperspherical coordinates
        hyperspherical = cartesian_to_hypersphere(centered)

        # Stretch simplex into hypersphere, no longer conserving the angles

        np.cos(np.pi / 3) / np.cos(2 / 3 * np.pi - angles)
        hyperspherical[:, 0] /= np.cos(np.pi / 3) * np.cos(2 / 3 * np.pi - hyperspherical[:, 1:])

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
