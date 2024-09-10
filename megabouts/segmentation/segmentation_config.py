from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from megabouts.segmentation.threshold import estimate_threshold_using_GMM
from megabouts.utils.utils_math import find_onset_offset_numpy
#from megabouts.config.component_configs import SegmentationConfig, TailSegmentationConfig, TrajSegmentationConfig
from typing import Tuple,Dict

from megabouts.pipeline.base_config import BaseConfig

@dataclass
class SegmentationConfig(BaseConfig):
    """Configuration for generic preprocessing.

    Attributes:
        bout_duration_ms (float): Duration of the bout in milliseconds.
        first_half_beat_loc_ms (float): Location of first half beat within a bout in milliseconds.
    """
    bout_duration_ms: float = 200
    first_half_beat_loc_ms: float = 36
    
    @property
    def bout_duration(self):
        """Convert the minimum segment size from milliseconds to frames."""
        return self.convert_ms_to_frames(self.bout_duration_ms)
    @property
    def first_half_beat_loc(self):
        """Convert the first half beat location from milliseconds to frames."""
        return self.convert_ms_to_frames(self.first_half_beat_loc_ms)
    
@dataclass
class TailSegmentationConfig(SegmentationConfig):
    """Configuration for tail segmentation.

    Attributes:
        long_rec_tail_speed_thresh_std (float): Standard deviation threshold for tail speed (used for long recording)
        short_rec_tail_speed_thresh (float):  Threshold for tail speed (used for short recording)
        min_bout_duration_ms (float): Minimum bout duration in milliseconds.
    """
    threshold_params: Dict = field(default_factory=dict)
    min_bout_duration_ms: float = 85

    def __post_init__(self):
        super().__post_init__()
        # Set default 'method' if not provided
        if 'method' not in self.threshold_params:
            self.threshold_params['method'] = 'GMM'
        if self.threshold_params.get('method') == 'GMM' and 'thresh_std' not in self.threshold_params:
            self.threshold_params['thresh_std'] = 2.1
        if self.threshold_params.get('method') == 'simple' and 'thresh' not in self.threshold_params:
            self.threshold_params['thresh'] = 1 

    @property
    def min_bout_duration(self):
        """Convert the minimum segment size from milliseconds to frames."""
        return self.convert_ms_to_frames(self.min_bout_duration_ms)

@dataclass
class TrajSegmentationConfig(SegmentationConfig):
    """Configuration for traj segmentation.

    Attributes:
        peak_prominence (float): prominence for peak finding algorithm
        margin_before_peak_ms (float): Margin from peak to onset in milliseconds.
    """
    peak_prominence: float = 0.4
    peak_percentage: float = 0.2
    #margin_before_peak_ms: float = 28

    #@property
    #def margin_before_peak(self):
    #    """Convert the minimum segment size from milliseconds to frames."""
    #    return self.convert_ms_to_frames(self.margin_before_peak_ms)
