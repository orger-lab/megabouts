import numpy as np
from scipy.interpolate import interp1d
from ..utils.math_utils import compute_angle_between_vectors


def interpolate_tail_keypoint(tail_x, tail_y, n_segments=10):
    """
    Interpolates the tail keypoints to create a curve with a specified number of segments.

    Parameters:
        tail_x (numpy.ndarray): The x-coordinates of the tail keypoints. Shape: (T, n_segments_input).
        tail_y (numpy.ndarray): The y-coordinates of the tail keypoints. Shape: (T, n_segments_input).
        n_segments (int, optional): The number of segments to interpolate. Default: 10.

    Returns:
        tuple: Interpolated x-coordinates and y-coordinates of the tail keypoints. Shape: (T, n_segments+1).
    """
    if n_segments < 2:
        raise ValueError("There should be more than 3 keypoints.")

    T, n_segments_init = tail_x.shape[0], tail_x.shape[1]
    tail_x_interp = np.full((T, n_segments + 1), np.nan)
    tail_y_interp = np.full((T, n_segments + 1), np.nan)

    for i in range(T):
        try:
            points = np.array([tail_x[i, :], tail_y[i, :]]).T
            is_nan = np.any(np.isnan(points))
            if not is_nan:
                id_first_nan = points.shape[0]
                N_seg = n_segments + 1
            else:
                id_first_nan = np.where(np.any(np.isnan(points), axis=1))[0][0]
                N_seg = int(np.round(id_first_nan / n_segments_init * (n_segments + 1)))

            alpha = np.linspace(0, 1, N_seg)
            distance = np.cumsum(
                np.sqrt(np.sum(np.diff(points[:id_first_nan, :], axis=0) ** 2, axis=1))
            )
            distance = np.insert(distance, 0, 0) / distance[-1]

            kind = "cubic" if len(distance) > 3 else "linear"
            interpolator = interp1d(
                distance, points[:id_first_nan, :], kind=kind, axis=0
            )

            curve = interpolator(alpha)
            tail_x_interp[i, :N_seg] = curve[:, 0]
            tail_y_interp[i, :N_seg] = curve[:, 1]
        except IndexError:
            # If interpolation fails, keep NaN values for this frame
            continue

    return tail_x_interp, tail_y_interp


def compute_angles_from_keypoints(head_x, head_y, tail_x, tail_y):
    """
    Computes the tail angles and body angle based on keypoints.

    Args:
        head_x (ndarray): X-coordinates of the head keypoints.
        head_y (ndarray): Y-coordinates of the head keypoints.
        tail_x (ndarray): X-coordinates of the tail keypoints.
        tail_y (ndarray): Y-coordinates of the tail keypoints.

    Returns:
        tuple: A tuple containing the tail angles (tail_angle) and the body angle (head_yaw).
    """

    if len(tail_x.shape) == 1:
        tail_x = tail_x[:, np.newaxis]
        tail_y = tail_y[:, np.newaxis]

    if tail_x.shape[1] != tail_y.shape[1]:
        raise ValueError("tail_x and tail_y must have same dimensions")

    T, N_keypoints = tail_x.shape[0], tail_x.shape[1]
    start_vector = np.vstack((tail_x[:, 0] - head_x, tail_y[:, 0] - head_y)).T
    head_yaw = np.arctan2(-start_vector[:, 1], -start_vector[:, 0])

    if N_keypoints > 1:
        vector_tail_segment = np.stack(
            (np.diff(tail_x, axis=1), np.diff(tail_y, axis=1)), axis=2
        )
        relative_angle = np.zeros((T, N_keypoints - 1))
        relative_angle[:, 0] = compute_angle_between_vectors(
            start_vector, vector_tail_segment[:, 0, :]
        )
        for i in range(vector_tail_segment.shape[1] - 1):
            relative_angle[:, i + 1] = compute_angle_between_vectors(
                vector_tail_segment[:, i, :], vector_tail_segment[:, i + 1, :]
            )
        tail_angle = np.cumsum(relative_angle, axis=1)
    else:
        tail_angle = None
    return tail_angle, head_yaw


def convert_tail_angle_to_keypoints(
    head_x, head_y, head_yaw, tail_angle, body_to_tail_mm=0.5, tail_to_tail_mm=0.32
):
    """
    Converts tail angles back to keypoints.

    Args:
        head_x (ndarray): X-coordinates of the head.
        head_y (ndarray): Y-coordinates of the head.
        head_yaw (ndarray): Body angle.
        tail_angle (ndarray): Tail angles.
        body_to_tail_mm (float): Distance from body to first tail keypoint in mm. Default: 0.5.
        tail_to_tail_mm (float): Distance between consecutive tail keypoints in mm. Default: 0.32.

    Returns:
        tuple: Tail keypoints (tail_x, tail_y).
    """
    T, num_segments = tail_angle.shape[0], tail_angle.shape[1] + 1
    tail_x = np.zeros((T, num_segments))
    tail_y = np.zeros((T, num_segments))

    for i in range(T):
        head_pos = np.array([head_x[i], head_y[i]])
        body_vect = np.array([np.cos(head_yaw[i]), np.sin(head_yaw[i])])
        swim_bladder = head_pos - body_vect * body_to_tail_mm

        tail_x[i, 0], tail_y[i, 0] = swim_bladder
        tail_angle_abs = tail_angle[i, :] + (head_yaw[i] + np.pi)
        tail_pos = swim_bladder

        for j in range(tail_angle.shape[1]):
            tail_vect = np.array([np.cos(tail_angle_abs[j]), np.sin(tail_angle_abs[j])])
            tail_pos += tail_to_tail_mm * tail_vect
            tail_x[i, j + 1] = tail_pos[0]
            tail_y[i, j + 1] = tail_pos[1]

    return tail_x, tail_y


def interpolate_tail_angle(tail_angle, n_segments=10):
    """
    Interpolates tail angles.

    Args:
        tail_angle (ndarray): Tail angles.
        n_segments (int, optional): Number of segments. Default: 10.

    Returns:
        ndarray: Interpolated tail angles.
    """
    T = tail_angle.shape[0]
    body_x, body_y, body_angle = np.zeros(T) + 0.5, np.zeros(T), np.zeros(T)
    tail_x, tail_y = convert_tail_angle_to_keypoints(
        body_x, body_y, body_angle, tail_angle
    )

    tail_x_interp, tail_y_interp = interpolate_tail_keypoint(
        tail_x, tail_y, n_segments=n_segments
    )
    tail_angle_interp, head_yaw = compute_angles_from_keypoints(
        body_x, body_y, tail_x_interp, tail_y_interp
    )
    return tail_angle_interp  # ,tail_x,tail_y,tail_x_interp, tail_y_interp
