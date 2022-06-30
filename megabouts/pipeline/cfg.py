from dataclasses import dataclass,field
from super_resolution.downsampling import convert_ms_to_frames
import numpy as np


@dataclass
class ConfigTrajPreprocess:
    """All parameters values relative to time should be in ms
    """
    fps: float
    freq_cutoff_min: float = 20
    beta: float = 1
    robust_diff_filt_ms: float = 21
    lag_mobility_ms : float = 85
    bout_duration_ms= float = 200
    @property
    def robust_diff(self):
        res = convert_ms_to_frames(self.fps,self.robust_diff_filt_ms)    
        if res%2==0:
            res=res+1 # Make sure robust_diff_dn is odd
        return res
    @property
    def lag_mobility(self):
        return convert_ms_to_frames(self.fps,self.lag_mobility_ms)  

@dataclass
class ConfigTrajSegmentationClassification:
    """All parameters values relative to time should be in ms
    """
    fps: float
    peak_prominence: float = 0.4
    margin_before_peak_ms: float = 28
    bout_duration_ms: float = 200
    augment_max_delay_ms: float = 28
    augment_step_delay_ms: float = 4
    feature_weight: np.ndarray = np.array([0.4,0.4,2])
    N_kNN: int = 50

    @property
    def margin_before_peak(self):
        return convert_ms_to_frames(self.fps,self.margin_before_peak_ms)        
    @property
    def bout_duration(self):
        return convert_ms_to_frames(self.fps,self.bout_duration_ms)        
    @property
    def augment_max_delay(self):
        return convert_ms_to_frames(self.fps,self.augment_max_delay_ms)        
    @property
    def augment_step_delay(self):
        return convert_ms_to_frames(self.fps,self.augment_step_delay_ms)     