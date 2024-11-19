import numpy as np
from scipy.interpolate import interp1d


def convert_frame_duration(n_frames_origin: int, fps_origin: int, fps_new: int) -> int:
    """Convert number of frames between different sampling rates.

    Parameters
    ----------
    n_frames_origin : int
        Number of frames in original time series
    fps_origin : int
        Original frames per second
    fps_new : int
        Target frames per second

    Returns
    -------
    int
        Number of frames at target fps

    Examples
    --------
    >>> convert_frame_duration(700, fps_origin=700, fps_new=100)
    100
    """

    return int(np.ceil(fps_new / fps_origin * n_frames_origin))


def convert_ms_to_frames(fps: int, duration: float) -> int:
    """Convert duration in milliseconds to number of frames.

    Parameters
    ----------
    fps : int
        Frame rate
    duration : float
        Duration in milliseconds

    Returns
    -------
    int
        Number of frames corresponding to duration

    Examples
    --------
    >>> convert_ms_to_frames(fps=100, duration=100)  # 100ms at 100fps
    10
    """
    n_frames = int(np.ceil(duration * fps / 1000))
    return n_frames


def create_downsampling_function(
    fps_new: int, fps_origin: int, duration: int, duration_unit="ms", kind="linear"
):
    """Generate function for downsampling time series data.

    Parameters
    ----------
    fps_new : int
        Target frames per second
    fps_origin : int
        Original frames per second
    duration : float, optional
        Duration of sequence
    duration_unit : str, optional
        Unit for duration: 'ms' or 'frames', by default 'ms'
    kind : str, optional
        Interpolation method: 'linear', 'nearest', 'cubic', etc., by default 'linear'

    Returns
    -------
    downsampling_f : callable
        Function that downsamples arrays along specified axis
    n_frames_new : int
        Number of frames in downsampled output
    t : ndarray
        Original time points
    tnew : ndarray
        Downsampled time points

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.rand(700, 10)  # 700fps data, 10 features
    >>> downsample_f, n_frames, t, tnew = create_downsampling_function(fps_new=350, fps_origin=700, duration=len(x), duration_unit='frames')
    >>> x_downsampled = downsample_f(x, axis=0)
    >>> x_downsampled.shape[0] == 350  # check output length
    True
    """
    duration_units = ["ms", "frames"]
    if duration_unit not in duration_units:
        raise ValueError("Invalid duration_unit. Expected one of: %s" % duration_units)

    if duration_unit == "ms":
        duration_ms = duration
        n_frames_original = convert_ms_to_frames(fps_origin, duration_ms)
        n_frames_new = convert_ms_to_frames(fps_new, duration_ms)
    else:
        n_frames_original = duration
        n_frames_new = convert_frame_duration(n_frames_original, fps_origin, fps_new)
        duration_ms = duration * 1000 / fps_origin

    t = np.linspace(0, duration_ms, n_frames_original, endpoint=False)
    tnew = np.linspace(0, duration_ms, n_frames_new, endpoint=False)

    def downsampling_f(x, axis=0):
        # Make the interpolator function.
        func = interp1d(t, x, kind=kind, axis=axis)
        xnew = func(tnew)

        return xnew

    return downsampling_f, n_frames_new, t, tnew
