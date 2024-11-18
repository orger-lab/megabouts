import numpy as np

from pybaselines.smooth import noise_median
from scipy import ndimage

from pybaselines.whittaker import asls
from scipy.signal import savgol_filter

from functools import partial


def compute_baseline(x, method, params):
    """Main function to compute baseline

    Args:
        x (np.ndarray): input signal (T,)
        method (string): can be either 'slow' 'beads' or 'whittaker'
        params (dict): dict of parameters. Should contains fps field.
        If 'whittaker' should containd field lambda.

    Returns:
        np.ndarray: baseline of x
    """

    fps = params["fps"]
    if method == "None":
        baseline = np.zeros_like(x)
    elif method == "median":
        window_size = 2 * params["half_window"] + 1
        baseline = ndimage.median_filter(x, size=window_size, mode="constant", cval=0)
        baseline = ndimage.gaussian_filter(baseline, sigma=window_size / 6)
        # baseline = np.zeros_like(x)
        """
        baseline = noise_median(x, 
                                half_window=params['half_window'], 
                                smooth_half_window=None, 
                                sigma=None,
                                mode='constant',
                                constant_values=0)[0]"""
    elif method == "whittaker":
        baseline_func = partial(
            compute_baseline_whittaker,
            win_slow=params["half_window"],
            win_std=int(fps / 2),
            thresh_sigma=1,
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


def compute_baseline_whittaker(
    x, win_slow=700, win_std=int(700 / 2), thresh_sigma=1, lmbda=1e4
):
    """
    Compute the baseline of a given input using the Whittaker smoother.

    Parameters:
    - x: 1D numpy array of the input values.
    - win_slow: int, slow window size to use when computing the slow baseline.
    - win_std: int, window size to use when computing the standard deviation for clipping.
    - thresh_sigma: float, number of standard deviations to use as the clipping threshold.
    - lmbda: float, regularization parameter to use in the Whittaker smoother.

    Returns:
    - baseline: 1D numpy array of the computed baseline.
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
    """
    Compute the baseline of a batch of input values using a given baseline computation function.

    Parameters:
    - x_batch: list of 1D numpy arrays, the input values to compute the baseline for.
    - baseline_func: function, the function to use to compute the baseline of each element of x_batch.

    Returns:
    - y_batch: list of 1D numpy arrays, the computed baselines for each element of x_batch.
    """
    y_batch = []
    for x in x_batch:
        y_batch.append(baseline_func(x))

    return y_batch


def compute_batch(x, L_center, L_edge):
    """Batch input into overlapping segments.
    The input will be mirror-padded to have a lenght divisible by L_center+L_edge.

    Args:
        x (np.ndarray): input time series to batch (T,)
        L_center (int): lenght of batch non overlapping
        L_edge (int): lenght of overlap between batch, should be larger than 2

    Returns:
        - x_batch - list of batched input including padding
        - left_edge - list of interval for the left side overlapping segments of each batch
        - center - list of interval for the center of each batch
        - right edge - list of interval for the right side overlapping segments of each batch
        - T - length of input x
        - T_padded - lenght of padded input x
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

    Args:
        y_batch (list): list of batched input including padding
        T_padded (int): length of padded input
        T (int): length of input before padding
        left_edge (list): list of intervals for left overlapping segments
        center (list): list of intervals for center segments
        right_edge (list): list of intervals for right overlapping segments
        L_center (int): length of non-overlapping batch segment
        L_edge (int): length of overlap between batches

    Returns:
        np.ndarray: merged signal of length T
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
