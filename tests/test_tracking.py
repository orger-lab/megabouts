import pytest
import numpy as np
from megabouts.tracking import (
    compute_tail_angles,
    # Add other tracking functions you want to test
)


def test_compute_tail_angles(sample_tail_points):
    """Test basic tail angle computation."""
    angles = compute_tail_angles(sample_tail_points)
    np.testing.assert_almost_equal(angles, [0.0, 0.46, 0.46], decimal=2)


def test_invalid_input():
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        compute_tail_angles(np.array([[0, 0]]))  # Too few points
