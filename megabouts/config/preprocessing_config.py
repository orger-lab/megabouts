from dataclasses import dataclass, field
from typing import Dict
from .base_config import BaseConfig
import numpy as np


@dataclass
class PreprocessingConfig(BaseConfig):
    """Configuration for generic preprocessing.

    Attributes:
        limit_na_ms (float): Limit for consecutive NA values to interpolate in milliseconds.
    """

    limit_na_ms: float = 100

    @property
    def limit_na(self) -> int:
        return self.convert_ms_to_frames(self.limit_na_ms)


@dataclass
class TrajPreprocessingConfig(PreprocessingConfig):
    """Configuration for head trajectory preprocessing.

    All parameters values relative to time should be in ms.

    Attributes:
        freq_cutoff_min (float): Minimum frequency cutoff for 1euro filter in Hz.
        beta (float): Beta value for 1euro filter.
        robust_diff_filt_ms (float): Filter size for robust difference in milliseconds.
        lag_kinematic_activity_ms (float): Lag for kinematic activity in milliseconds.
    """

    freq_cutoff_min: float = 20
    beta: float = 1
    robust_diff_filt_ms: float = 21
    lag_kinematic_activity_ms: float = 85

    @property
    def robust_diff(self):
        res = self.convert_ms_to_frames(self.robust_diff_filt_ms)
        if res % 2 == 0:
            res += 1  # Ensure robust_diff is odd
        return max(res, 3)

    @property
    def lag_kinematic_activity(self):
        return self.convert_ms_to_frames(self.lag_kinematic_activity_ms)


@dataclass
class TailPreprocessingConfig(PreprocessingConfig):
    """Configuration for tail preprocessing.

    All parameters values relative to time should be in ms.

    Attributes:
        num_pcs (int): Number of principal components.
        savgol_window_ms (float): Window size for Savitzky-Golay filter in milliseconds.
        baseline_method (str): Method for baseline computation.
        baseline_params (Dict): Parameters for baseline computation.
        tail_speed_filter_ms (float): Filter size for tail speed in milliseconds.
        tail_speed_boxcar_filter_ms (float): Boxcar filter size for tail speed in milliseconds.
    """

    num_pcs: int = 4
    savgol_window_ms: float = 15
    baseline_method: str = "median"
    baseline_params: Dict = field(default_factory=dict)
    tail_speed_filter_ms: float = 100
    tail_speed_boxcar_filter_ms: float = 14

    def __post_init__(self):
        super().__post_init__()
        self.baseline_params["fps"] = self.fps
        self.baseline_params["half_window"] = int(np.round(self.fps / 2))

    @property
    def savgol_window(self) -> int:
        res = self.convert_ms_to_frames(self.savgol_window_ms)
        if res % 2 == 0:
            res += 1  # Ensure savgol_window is odd
        return res

    @property
    def tail_speed_filter(self) -> int:
        n = self.convert_ms_to_frames(self.tail_speed_filter_ms)
        if n % 2 == 0:
            n += 1
        return n

    @property
    def tail_speed_boxcar_filter(self) -> int:
        n = self.convert_ms_to_frames(self.tail_speed_boxcar_filter_ms)
        if n == 0:
            return 1
        else:
            return n
