"""The functions for converting to and from the Hyperbolic simplex basis"""
from copy import deepcopy
import logging

import numpy as np

#from psych_metric.distrib.simplex.euclidean import EuclideanSimplexTransform


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


def barycentric_to_cartesian():
    pass


def givens_rotation(dim, x, y, angle):
    """Creates a Givens rotation matrix."""
    rotate = np.eye(dim)
    rotate[x, x] = np.cos(angle)
    #rotate[x, y] = -np.sin(angle)
    rotate[x, y] = np.sin(angle)
    rotate[y, x] = -np.sin(angle)
    rotate[y, y] = np.cos(angle)
    return rotate


def rotate_around(rotation_simplex, angle):
    """General n-dimension rotation."""
    if rotation_simplex.shape[0] > rotation_simplex.shape[0]:
        # expects points to contained in the rows, elements in columns
        rotation_simplex = rotation_simplex.T

    # Transpose the simplex st the first point is centered at origin
    v = rotation_simplex - rotation_simplex[0]

    n = rotation_simplex.shape[1]
    mat = np.eye(n)

    k = 0
    print(f'k = {k}')

    print(f'v:\n{v}')
    print(f'mat:\n{mat}')

    for r in range(1, n-1):
        print(f'\nr = {r}, from [1, {n-1}).')
        for c in list(range(r, n))[::-1]:
            print(f'\t-----\n\tc = {c}, from ({n}, {r}].')
            k += 1
            print(f'\tk = {k}')
            print(f'rot k={k}\'s angle = {np.arctan2(v[r, c], v[r, c - 1])}')
            rot = givens_rotation(
                n,
                c,
                c - 1,
                np.arctan2(v[r, c], v[r, c - 1]),
            )
            v = v @ rot
            mat = mat @ rot

            print(f'\trot:\n{rot}')
            print(f'\tv:\n{v}')
            print(f'\tmat:\n{mat}')

    return (
        rotation_simplex[0],
        mat @ givens_rotation(n, n - 2, n - 1, angle) @ np.linalg.inv(mat),
    )


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

    # TODO this use Barycentric Coordinates

    # after rotation about arbitary n-2 space, go into Barycentric Coordinates
    # cartesian to barycentric

    # for every angle set given, find the minimum Barycentric coord and zero it
    # while increasing the remaining coordinates approriately, thus getting the
    # point on the d-1 simplex opposite of the zeroed coord.

    # convert that back into Cartesian and get the radius (L2norm)
    # barycentric to cartesian

    # that is the simplex's boundary point's radius at the given angle set
    return


class Rotator(object):
    """Class the contains the process for rotating about some n-2 space."""
    def __init__(self, rotation_simplex, angle):
        self.translate, self.rotate_drop_dim = rotate_around(
            rotation_simplex,
            angle,
        )

    def rotate(self, vectors, drop_dim=False):
        """Rotates the vectors of the n-1 simplex from n dimensions to n-1
        dimensions.
        """
        # TODO expects shape of 2, add check on vectors
        result = (vectors - self.translate) @ self.rotate_drop_dim \
            + self.translate
        if drop_dim:
            return result[:, 1:]
        return result

    def inverse(self, vectors):
        """Rotates the vectors of the n-1 simplex from n-1 dimensions to n
        dimensions.
        """
        # TODO expects shape of 2, add check on vectors
        if vectors.shape[1] == len(self.translate):
            return (vectors - self.translate) \
                @ np.linalg.inv(self.rotate_drop_dim) + self.translate
        return (
            (
                np.hstack((np.zeros([len(vectors), 1]), vectors))
                - self.translate
            )
            @ np.linalg.inv(self.rotate_drop_dim)
            + self.translate
        )


class EuclideanSimplexTransform(object):
    """Creates and contains the objects needed to convert to and from the
    Euclidean Simplex basis.

    Attributes
    ----------
    cart_simplex : np.ndarray
    centroid : np.ndarray
    rotator : Rotator
        Used to find cart_simplex and for reverse transforming from the
        cartesian simplex to the original probability simplex.

    Properties
    ----------
    input_dim : int
        The number of dimensions of the input samples before being transformed.
    output_dim : int

    Note
    ----
    This EuclideanSimplexTransform takes more time and more memory than the
    original that used QR or SVD to find the rotation matrix. However, this
    version preserves the simplex dimensions, keeping the simplex regular,
    while the QR and SVD found rotation matrices do not.
    """
    def __init__(self, dim):
        prob_simplex_verts = np.eye(dim)

        # Get the angle to rotate about the n-2 space to zero out first dim
        angle_to_rotate = -np.arctan2(
            1.0,
            np.linalg.norm([1 / (dim - 1)] * (dim - 1)),
        )

        # Rotate to zero out one arbitrary dimension, drop that zeroed dim.
        self.rotator = Rotator(prob_simplex_verts[1:], angle_to_rotate)
        self.cart_simplex = self.rotator.rotate(prob_simplex_verts)

        # Center Simplex in (N-1)-dim (find centroid and adjust via that)
        self.centroid = np.mean(self.cart_simplex, axis=0)
        self.cart_simplex -= self.centroid

        # Save the vertices of the rotated simplex, transposed for ease of comp
        self.cart_simplex = self.cart_simplex.T
        # TODO Decide if keeping the cart_simplex for going from prob simplex
        # to cart simplex with only one matrix multiplication is worth keeping
        # the (n,n) matrix.

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        for k, v in self.__dict__.items():
            setattr(new, k, deepcopy(v, memo))

        return new

    @property
    def input_dim(self):
        # TODO wrap all code for obtaining cart_simllex in EuclideanTransform
        #return self.euclid_simplex_transform.input_dim
        return self.cart_simplex.shape[0]

    @property
    def output_dim(self):
        #return self.euclid_simplex_transform.output_dim
        return self.cart_simplex.shape[1]

    def to(self, vectors, drop_dim=True):
        """Transform given vectors into hyperbolic probability simplex space."""
        # Convert from probability simplex to the Cartesian coordinates of the
        # centered, regular simplex
        if drop_dim:
            return (vectors @ self.cart_simplex)[:, 1:]
        return vectors @ self.cart_simplex

    def back(self, vectors):
        """Transform given vectors into hyperbolic probability simplex space."""
        # Convert from probability simplex to the Cartesian coordinates of the
        # centered, regular simplex
        #aligned = vectors @ self.aligned_simplex.T

        # TODO expects shape of 2, add check on vectors
        if vectors.shape[1] == self.input_dim:
            return self.rotator.inverse(vectors + self.centroid)
        return self.rotator.inverse(vectors + self.centroid)


class HyperbolicSimplexTransform(object):
    """

    Attributes
    ----------
    """

    def __init__(self, dim):
        self.est = EuclideanSimplexTransform(dim)

        # Save the radius of the simplex's circumscribed hypersphere
        self.circumscribed_radius = np.linalg.norm(self.est.cart_simplex[:, 0])

    @property
    def input_dim(self):
        # TODO wrap all code for obtaining cart_simllex in EuclideanTransform
        return self.est.input_dim

    @property
    def output_dim(self):
        return self.est.output_dim

    def to(self, vectors):
        """Transform given vectors into hyperbolic probability simplex space."""
        # Convert from probability simplex to the Cartesian coordinates of the
        # centered, regular simplex
        #aligned = vectors @ self.aligned_simplex.T
        aligned = self.est.to(vectors, drop_dim=True)

        # Convert to polar/hyperspherical coordinates
        hyperspherical = cartesian_to_hypersphere(aligned)

        # TODO Stretch simplex into hypersphere, no longer conserving the angles


        # convert each pt to Barycentric coordinates
        # select minimum coord as dim to zero to get boundary pt on d simplex
        # get boundary points radius
        # scale each point by * circum_radius / simplex_boundary_radius

        #ok = np.cos(np.pi / 3) / np.cos(2 / 3 * np.pi - angles)
        #hyperspherical[:, 0] /= np.cos(np.pi / 3) * np.cos(2 / 3 * np.pi - hyperspherical[:, 1:])

        # TODO Inverse Poincare' Ball method to go into hyperbolic space

        return

    def tf_to(self, vectors):
        """Transform given vectors into n-1 probability simplex space done in
        tensorflow code.
        """
        return

    def back(self, vectors):
        """Transform given vectors out of n-1 probability simplex space."""
        # Poinecare's Ball

        # Circumscribed hypersphere to Cartesian simplex

        # Cartesian simplex to probability distribution (Barycentric coord)
        return self.est.back(vectors)

    def tf_from(self, vectors):
        """Transform given vectors out of n-1 probability simplex space done in
        tensorflow code.
        """
        return
