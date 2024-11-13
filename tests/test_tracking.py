import numpy as np
from megabouts.utils.math_utils import compute_angle_between_vectors


def test_compute_angle_between_vectors():
    # Test case 1: Perpendicular vectors (90 degrees)
    v1 = np.array([[1, 0]])
    v2 = np.array([[0, 1]])
    angle = compute_angle_between_vectors(v1, v2)
    assert np.isclose(angle, np.pi / 2)

    # Test case 2: Same direction (0 degrees)
    v1 = np.array([[1, 0]])
    v2 = np.array([[1, 0]])
    angle = compute_angle_between_vectors(v1, v2)
    assert np.isclose(angle, 0)

    # Test case 3: Opposite direction (180 degrees)
    v1 = np.array([[1, 0]])
    v2 = np.array([[-1, 0]])
    angle = compute_angle_between_vectors(v1, v2)
    assert np.isclose(abs(angle), np.pi)
