import numpy as np

from pybaselines.smooth import noise_median
from scipy import ndimage

from pybaselines.whittaker import asls
from scipy.signal import savgol_filter

from functools import partial


def compute_baseline(x, method, params):
    """Main function to compute baseline.

    Parameters
    ----------
    x : np.ndarray
        Input signal, shape (T,)
    method : str
        Method for baseline computation:
        - 'None': returns zero baseline
        - 'median': uses median filter with gaussian smoothing
        - 'whittaker': uses Whittaker smoother with adaptive weights
    params : dict
        Parameters for baseline computation. Must contain:
        - 'fps': sampling rate
        - 'half_window': for median method
        - 'lmbda': for whittaker method

    Returns
    -------
    np.ndarray
        Baseline of x, same shape as input

    Notes
    -----
    For 'whittaker' method with sequences longer than 100*fps frames,
    the computation is done in batches for efficiency.
    """

    fps = params["fps"]
    if method == "None":
        baseline = np.zeros_like(x)
    elif method == "median":
        window_size = 2 * params["half_window"] + 1
        baseline = ndimage.median_filter(x, size=window_size, mode="constant", cval=0)
        baseline = ndimage.gaussian_filter(baseline, sigma=window_size / 6)

    elif method == "whittaker":
        baseline_func = partial(
            compute_baseline_whittaker,
            win_slow=params["half_window"],
            lmbda=params["lmbda"],
        )
    else:
        raise ValueError(f"Unsupported value for `method`: {method}")

    if method == "whittaker":
        if (len(x) > fps * 100) & (len(x) > 30000):
            L_center = fps * 28
            L_edge = fps
            x_batch, left_edge, center, right_edge, T, T_padded = compute_batch(
                x, L_center, L_edge
            )
            y_batch = compute_baseline_on_batch(x_batch, baseline_func)
            baseline = merge_batch(
                y_batch, T_padded, T, left_edge, center, right_edge, L_center, L_edge
            )
            # Further filtering to remove tiny oscillation:
            L_ms = 144
            L = int(np.ceil(L_ms * fps / 1000))
            L = L + 1 if L % 2 == 0 else L
            baseline = savgol_filter(baseline, L, 3)
        else:
            baseline = compute_baseline_on_batch([x], baseline_func)[0]
    return baseline


def compute_baseline_whittaker(x, win_slow=700, lmbda=1e4):
    """Compute the baseline of a given input using the Whittaker smoother.

    Parameters
    ----------
    x : np.ndarray
        Input signal, shape (T,)
    win_slow : int, optional
        Window size for slow baseline computation, by default 700
    lmbda : float, optional
        Regularization parameter for Whittaker smoother, by default 1e4

    Returns
    -------
    np.ndarray
        Baseline signal, same shape as input
    """
    slow_baseline = noise_median(
        x,
        half_window=win_slow,
        smooth_half_window=None,
        sigma=None,
        mode="constant",
        constant_values=0,
    )[0]

    x_centered = x - slow_baseline

    # Compute baseline of signal directly without clipping:
    fast_baseline = asls(
        x_centered, lam=lmbda, p=0.5, diff_order=2, max_iter=50, tol=0.001, weights=None
    )[0]

    return slow_baseline + fast_baseline


def compute_baseline_on_batch(x_batch, baseline_func):
    """Compute baseline for each element in a batch.

    Parameters
    ----------
    x_batch : list
        List of 1D numpy arrays to compute baseline for
    baseline_func : callable
        Function to compute baseline for each array

    Returns
    -------
    list
        List of computed baselines for each input array
    """
    y_batch = []
    for x in x_batch:
        y_batch.append(baseline_func(x))

    return y_batch


def compute_batch(x, L_center, L_edge):
    """Batch input into overlapping segments.

    The input will be mirror-padded to have a length divisible by L_center+L_edge.

    Parameters
    ----------
    x : np.ndarray
        Input time series to batch, shape (T,)
    L_center : int
        Length of non-overlapping batch segment
    L_edge : int
        Length of overlap between batches, should be larger than 2

    Returns
    -------
    x_batch : list
        List of batched input including padding
    left_edge : list
        Intervals for left overlapping segments
    center : list
        Intervals for center segments
    right_edge : list
        Intervals for right overlapping segments
    T : int
        Length of input x
    T_padded : int
        Length of padded input x
    """
    T = len(x)
    T_padded = int(np.ceil(T / (L_edge + L_center)) * (L_center + L_edge))

    # Mirror x end to make it lenght T_padded
    x_padded = np.zeros(T_padded)
    x_padded[: len(x)] = x
    T_diff = len(x_padded) - len(x)
    x_padded[len(x) + 1 :] = x[:-T_diff:-1]

    # Batching:
    left_edge = [[0, 0]]
    center = [[0, L_center]]
    right_edge = [[L_center, L_center + L_edge]]

    n = int(T_padded / (L_center + L_edge))
    for i in range(1, n):
        left_edge.append(right_edge[-1])
        center.append([right_edge[-1][1], right_edge[-1][1] + L_center])
        right_edge.append([center[-1][1], center[-1][1] + L_edge])

    # Compute batch:
    x_batch = []
    for i in range(len(center)):
        interval = (left_edge[i][0], right_edge[i][1])
        x_batch.append(x_padded[interval[0] : interval[1]])

    return x_batch, left_edge, center, right_edge, T, T_padded


def merge_batch(y_batch, T_padded, T, left_edge, center, right_edge, L_center, L_edge):
    """Unbatch input signal into a time series.

    Uses sigmoid weighting to smoothly merge overlapping segments.

    Parameters
    ----------
    y_batch : list
        List of batched input including padding
    T_padded : int
        Length of padded input
    T : int
        Length of input before padding
    left_edge : list
        Intervals for left overlapping segments
    center : list
        Intervals for center segments
    right_edge : list
        Intervals for right overlapping segments
    L_center : int
        Length of non-overlapping batch segment
    L_edge : int
        Length of overlap between batches

    Returns
    -------
    np.ndarray
        Merged signal of length T
    """
    # Pre-compute sigmoid weights
    x = np.linspace(-L_edge / 2, L_edge / 2, L_edge)
    sigma = L_edge / 7
    weights = 1 / (1 + np.exp(-x / sigma))
    inv_weights = 1 - weights

    y = np.zeros(T_padded)
    n_batches = len(y_batch)

    # Handle first batch
    y[center[0][0] : center[0][1]] = y_batch[0][0:L_center]
    y[right_edge[0][0] : right_edge[0][1]] = (
        y_batch[0][L_center:] * inv_weights + y_batch[1][0:L_edge] * weights
    )

    # Handle middle batches
    for i in range(1, n_batches - 1):
        # Merge left edge
        y[left_edge[i][0] : left_edge[i][1]] = (
            y_batch[i - 1][-L_edge:] * inv_weights + y_batch[i][0:L_edge] * weights
        )
        # Copy center directly
        y[center[i][0] : center[i][1]] = y_batch[i][L_edge : L_edge + L_center]
        # Merge right edge
        y[right_edge[i][0] : right_edge[i][1]] = (
            y_batch[i][L_center + L_edge :] * inv_weights
            + y_batch[i + 1][0:L_edge] * weights
        )

    # Handle last batch
    y[left_edge[-1][0] : left_edge[-1][1]] = (
        y_batch[-2][L_center + L_edge :] * inv_weights + y_batch[-1][:L_edge] * weights
    )
    y[center[-1][0] : right_edge[-1][1]] = y_batch[-1][L_edge:]

    return y[:T]
