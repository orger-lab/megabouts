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
    """Compute robust derivative using Holoborodko's smooth noise-robust differentiator.

    Uses a central difference scheme optimized for noise reduction while preserving
    the signal's higher derivatives. See:
    https://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/

    Parameters
    ----------
    x : ndarray
        Input signal, shape (n_samples,)
    dt : float, optional
        Time step between samples, by default 1/700
    filter_length : int, optional
        Length of the filter, must be odd, by default 71
        Longer filters give smoother derivatives but require more edge padding

    Returns
    -------
    ndarray
        Filtered derivative, shape (n_samples,)
        First and last (filter_length-1)/2 points are NaN

    Examples
    --------
    >>> t = np.linspace(0, 1, 701)  # 700 fps
    >>> x = np.sin(2*np.pi*t)  # sine wave
    >>> dx = robust_diff(x)
    >>> # Check derivative at middle point close to expected cos(2*pi*t)
    >>> np.abs(dx[350] - 2*np.pi*np.cos(2*np.pi*0.5)) < 0.1
    True
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
    """Computes the signed angle between two vectors.

    Parameters
    ----------
    v1 : ndarray
        First set of vectors with shape (num_vectors, 2)
    v2 : ndarray
        Second set of vectors with shape (num_vectors, 2)

    Returns
    -------
    ndarray
        Signed angles between vectors in radians, shape (num_vectors,)
        Positive angle indicates counterclockwise rotation from v1 to v2

    Examples
    --------
    >>> v1 = np.array([[1, 0], [0, 1]])  # right, up
    >>> v2 = np.array([[0, 1], [-1, 0]])  # up, left
    >>> angles = compute_angle_between_vectors(v1, v2)
    >>> np.allclose(angles, [np.pi/2, np.pi/2])
    True
    """
    dot_product = np.einsum("ij,ij->i", v1, v2)
    cos_angle = dot_product
    sin_angle = np.cross(v1, v2)
    angle = np.arctan2(sin_angle, cos_angle)

    return angle


def compute_outer_circle(x, y, interval=100):
    """Compute smallest circle enclosing a subset of points.

    Parameters
    ----------
    x : ndarray
        X coordinates of points
    y : ndarray
        Y coordinates of points
    interval : int, optional
        Sample points every `interval` steps, by default 100

    Returns
    -------
    tuple
        (xc, yc, radius) : Center coordinates and radius of enclosing circle

    Examples
    --------
    >>> theta = np.linspace(0, 2*np.pi, 1000)
    >>> x = np.cos(theta)  # unit circle
    >>> y = np.sin(theta)
    >>> xc, yc, r = compute_outer_circle(x, y)
    >>> np.allclose([xc, yc], [0, 0], atol=0.1)  # center near origin
    True
    >>> np.abs(r - 1.0) < 0.1  # radius near 1
    True
    """
    p = [(x[i], y[i]) for i in np.arange(0, len(x), interval)]
    Circle = smallestenclosingcircle.make_circle(p)
    xc = Circle[0]
    yc = Circle[1]
    radius = Circle[2]
    return (xc, yc, radius)
