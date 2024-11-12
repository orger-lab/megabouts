import pytest
import glob
from pathlib import Path

NOTEBOOKS = list(Path(".").glob("tutorial_*.ipynb"))

@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_notebook_runs_without_errors(notebook):
    """Test that the notebook runs without errors."""
    pytest.importorskip("nbval")
    pytest.nbval.plugin.pytest_collect_file(notebook)