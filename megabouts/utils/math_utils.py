import numpy as np
from scipy.special import binom
from megabouts.utils import smallestenclosingcircle


def find_onset_offset_numpy(binary_serie):
    """
    Find the onset, offset, and duration of runs of 1s in a binary sequence.

    Parameters:
    - binary_serie: 1D numpy array of binary values.

    Returns:
    - onset: 1D numpy array of the indices at which each run of 1s starts.
    - offset: 1D numpy array of the indices at which each run of 1s ends.
    - duration: 1D numpy array of the duration of each run of 1s in number of time steps.
    """
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(binary_serie, 1).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    onset = ranges[:, 0]
    offset = ranges[:, 1]
    duration = offset - onset
    return onset, offset, duration


def robust_diff(x, dt=1 / 700, filter_length=71):
    """
    Compute the robust estimate of the derivative of a sequence of values.

    Parameters:
    - x: 1D numpy array of values to differentiate.
    - dt: float, time step to use in the derivative calculation.
    - filter_length: int, length of the filter to use in the derivative calculation.

    Returns:
    - filtered: 1D numpy array of the robust derivative estimate of the input sequence.
    """
    if not filter_length % 2 == 1:
        raise ValueError("Filter length must be odd.")
    M = int((filter_length - 1) / 2)
    m = int((filter_length - 3) / 2)
    coefs = [
        (1 / 2 ** (2 * m + 1)) * (binom(2 * m, m - k + 1) - binom(2 * m, m - k - 1))
        for k in range(1, M + 1)
    ]
    coefs = np.array(coefs)
    kernel = np.concatenate((coefs[::-1], [0], -coefs))
    filtered = np.convolve(kernel, x, mode="valid")
    filtered = (1 / dt) * filtered
    filtered = np.concatenate((np.nan * np.ones(M), filtered, np.nan * np.ones(M)))
    return filtered


def compute_angle_between_vectors(v1, v2):
    """
    Computes the angle between two vectors.

    Args:
        v1 (ndarray): First set of vectors with shape (num_vectors, num_dimensions).
        v2 (ndarray): Second set of vectors with shape (num_vectors, num_dimensions).

    Returns:
        ndarray: Array of angles between the vectors.
    """
    dot_product = np.einsum("ij,ij->i", v1, v2)
    cos_angle = dot_product
    sin_angle = np.cross(v1, v2)
    angle = np.arctan2(sin_angle, cos_angle)

    return angle


def compute_outer_circle(x, y, interval=100):
    """
    Compute the smallest circle that encloses a set of points.

    The set of points is obtained by selecting every `interval`-th point from the input x and y sequences.

    Parameters:
    - x: 1D numpy array of x-coordinates of the points.
    - y: 1D numpy array of y-coordinates of the points.
    - interval: int, the interval at which to select points from the input sequences.

    Returns:
    - circle: tuple of (xc, yc, radius), where xc and yc are the coordinates of the center of the circle, and radius is the circle radius.
    """
    p = [(x[i], y[i]) for i in np.arange(0, len(x), interval)]
    Circle = smallestenclosingcircle.make_circle(p)
    xc = Circle[0]
    yc = Circle[1]
    radius = Circle[2]
    return (xc, yc, radius)
