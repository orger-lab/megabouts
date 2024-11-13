from dataclasses import dataclass
import os
import numpy as np
from megabouts.config.base_config import BaseConfig
from megabouts.utils.time_utils import create_downsampling_function


@dataclass
class SparseCodingConfig(BaseConfig):
    """
    Configuration for convolutional sparse coding.

    Attributes:
        Dict: numpy array of weights for the sparse code.
        lmbda: float, regularization parameter to use in the sparse coding optimization.
        gamma: float, weight to use for the inhibition term in the optimization.
        mu: float, weight to use for the L1 norm in the optimization.
        window_inhib_ms: float, minimum segment size in milliseconds.
        dict_peak_ms: float, location of the first half beat within a bout in milliseconds.
        N_atoms: int, number of atoms in the dictionary.
    """

    # Dict: np.ndarray = field(default_factory=lambda: np.zeros(1))
    lmbda: float = 0.01
    gamma: float = 0.01
    mu: float = 0.05
    window_inhib_ms: float = 85
    dict_peak_ms: float = 28
    vigor_win_ms: float = 30

    def __post_init__(self):
        self.load_dictionary()

    def load_dictionary(self):
        config_dir = os.path.dirname(__file__)
        sparse_coding_dir = os.path.join(os.path.dirname(config_dir), "sparse_coding")
        filename = os.path.join(sparse_coding_dir, "4atomsDictHR_allTensor.npy")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Dictionary file not found: {filename}")

        with open(filename, "rb") as handle:
            Dict_700fps = np.load(handle)

        downsampling_f, _, _, _ = create_downsampling_function(
            self.fps, 700, 140 * 1000 / 700
        )
        self.Dict = downsampling_f(Dict_700fps, axis=0)
        self.N_atoms = self.Dict.shape[2]

    @property
    def window_inhib(self):
        return self.convert_ms_to_frames(self.window_inhib_ms)

    @property
    def dict_peak(self):
        return self.convert_ms_to_frames(self.dict_peak_ms)

    @property
    def vigor_win(self):
        return self.convert_ms_to_frames(self.vigor_win_ms)
