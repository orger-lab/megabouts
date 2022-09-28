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
        res = max(res,3)
        return res
    @property
    def lag_mobility(self):
        return convert_ms_to_frames(self.fps,self.lag_mobility_ms)  
    
    

@dataclass
class ConfigTailPreprocess:
    """All parameters values relative to time should be in ms
    """
    fps: float
    num_pcs: int = 4
    limit_na_ms: float = 100
    baseline_method: str = 'slow'
    baseline_params: dict = field(default_factory=dict)
 
    @property
    def limit_na(self):
        return convert_ms_to_frames(self.fps,self.limit_na_ms)        
    

@dataclass
class ConfigSparseCoding:
    """All parameters values relative to time should be in ms
    """
    fps: float
    Dict: np.ndarray
    lmbda: float = 0.01
    gamma: float = 0.01    
    mu: float = 0.05
    window_inhib_ms: float = 85
    
    @property
    def window_inhib(self):
        return convert_ms_to_frames(self.fps,self.window_inhib_ms)        
    


@dataclass
class ConfigTailSegmentationClassification:
    """All parameters values relative to time should be in ms
    """
    fps: float
    min_code_height: float = 1
    min_spike_dist_ms: float = 200
    margin_before_peak_ms: float = 0
    bout_duration_ms: float = 200
    augment_max_delay_ms: float = 28
    augment_step_delay_ms: float = 4
    feature_weight: np.ndarray = np.ones(7)
    N_kNN: int = 50

    @property
    def min_spike_dist(self):
        return convert_ms_to_frames(self.fps,self.min_spike_dist_ms)         
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
    N_kNN: int = 15

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