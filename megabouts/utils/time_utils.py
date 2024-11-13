import numpy as np
from scipy.interpolate import interp1d


def convert_frame_duration(n_frames_origin: int, fps_origin: int, fps_new: int) -> int:
    """_summary_

    Args:
        n_frames (int): number of frames in input times series
        fps_origin (int): fps of input times series
        fps_new (int): target fps

    Returns:
        int: number of frames corresponding for target fps
    """

    return int(np.ceil(fps_new / fps_origin * n_frames_origin))


def convert_ms_to_frames(fps: int, duration: float) -> int:
    """Convert duration in ms to number of frames

    Args:
        fps (int): frame rate
        duration (float): duration in ms

    Returns:
        int: number of frames correponding to duration
    """
    n_frames = int(np.ceil(duration * fps / 1000))
    return n_frames


def create_downsampling_function(
    fps_new: int, fps_origin: int, duration=200, duration_unit="ms", kind="linear"
):
    """Generate function that will downsample according to fps_new

    Args:
        fps_new (int): target fps
        fps_origin (int, optional): Defaults to 700.
        duration (float): duration of sequence in ms or frames
        duration_units (string): 'ms' or 'frames'
        kind (str,optional): Defauts to linear, The string can be  'nearest', 'slinear', 'quadratic', 'cubic'...

    Returns:
        _type_: downsampling function that downsample a np.ndarray along a given axis
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
