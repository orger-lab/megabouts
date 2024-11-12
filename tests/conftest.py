import pytest
import numpy as np

@pytest.fixture
def sample_tail_points():
    """Provide minimal sample tail tracking points."""
    return np.array([
        [0, 0],
        [1, 0],
        [2, 1],
        [3, 2]
    ])

@pytest.fixture
def sample_trajectory():
    """Provide minimal sample trajectory."""
    return np.array([
        [0, 0],
        [1, 1],
        [2, 2]
    ])

@pytest.fixture(scope="session")
def test_data_path():
    """Return path to test data directory."""
    from pathlib import Path
    return Path(__file__).parent / "data"