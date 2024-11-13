import pytest
from pathlib import Path


def pytest_collect_file(parent, file_path: Path):
    """Add notebook files to pytest collection."""
    if file_path.suffix == ".ipynb":
        return pytest.Module.from_parent(parent, path=file_path)


def pytest_configure(config):
    """Configure pytest to include both tests and notebooks."""
    # Add nbmake option if not already present
    if "--nbmake" not in config.invocation_params.args:
        config.option.nbmake = True

    # Ensure verbose output
    config.option.verbose = True
