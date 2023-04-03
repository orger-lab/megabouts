import os
from pathlib import Path
from dataclasses import dataclass,field
import pickle
import numpy as np
from megabouts.utils.utils_downsampling import convert_ms_to_frames


@dataclass
class ConfigTrajPreprocess:
    """All parameters values relative to time should be in ms
    """
    fps: float
    freq_cutoff_min: float = 20
    beta: float = 1
    robust_diff_filt_ms: float = 21
    lag_kinematic_activity_ms : float = 85
    bout_duration_ms= float = 200
    @property
    def robust_diff(self):
        res = convert_ms_to_frames(self.fps,self.robust_diff_filt_ms)    
        if res%2==0:
            res=res+1 # Make sure robust_diff_dn is odd
        res = max(res,3)
        return res
    @property
    def lag_kinematic_activity(self):
        return convert_ms_to_frames(self.fps,self.lag_kinematic_activity_ms)  

@dataclass
class ConfigTailPreprocess:
    """All parameters values relative to time should be in ms
    """
    fps: float
    num_pcs: int = 4
    limit_na_ms: float = 100
    savgol_window_ms: float = 15
    baseline_method: str = 'None'
    baseline_params: dict = field(default_factory=dict)
    @property
    def limit_na(self):
        return convert_ms_to_frames(self.fps,self.limit_na_ms)        
    @property
    def savgol_window(self):
        res = convert_ms_to_frames(self.fps,self.savgol_window_ms)    
        if res%2==0:
            res=res+1 # Make sure robust_diff_dn is odd
        if res<3:
            res=-1
        return convert_ms_to_frames(self.fps,self.savgol_window_ms)        
        
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
    dict_peak_ms: float = 28
    
    @property
    def window_inhib(self):
        return convert_ms_to_frames(self.fps,self.window_inhib_ms)        
    
    @property
    def dict_peak(self):
        return convert_ms_to_frames(self.fps,self.dict_peak_ms)        

@dataclass
class ConfigTailSegmentation:
    """All parameters values relative to time should be in ms
    """
    fps: float
    tail_speed_filter_ms: float = 100
    tail_speed_boxcar_filter_ms: float = 14
    tail_speed_thresh_std: float = 2.1
    min_bout_duration_ms: float = 85
    margin_before_peak_ms: float = 28
    bout_duration_ms: float = 200

    @property
    def tail_speed_filter(self):
        n = convert_ms_to_frames(self.fps,self.tail_speed_filter_ms)   
        if n%2==0: n = n+1   
        return  n
    @property
    def tail_speed_boxcar_filter(self):
        return convert_ms_to_frames(self.fps,self.tail_speed_boxcar_filter_ms)        
    @property
    def min_bout_duration(self):
        return convert_ms_to_frames(self.fps,self.min_bout_duration_ms)   
    @property
    def margin_before_peak(self):
        return convert_ms_to_frames(self.fps,self.margin_before_peak_ms)        
    @property
    def bout_duration(self):
        return convert_ms_to_frames(self.fps,self.bout_duration_ms)

@dataclass
class ConfigTailSegmentationFromSparseCode:
    """All parameters values relative to time should be in ms
    """
    fps: float
    peak_prominence: float = 0.4
    min_code_height: float = 1
    min_spike_dist_ms: float = 200
    margin_before_peak_ms: float = 28
    bout_duration_ms: float = 200

    @property
    def min_spike_dist(self):
        return convert_ms_to_frames(self.fps,self.min_spike_dist_ms)         
    @property
    def margin_before_peak(self):
        return convert_ms_to_frames(self.fps,self.margin_before_peak_ms)        
    @property
    def bout_duration(self):
        return convert_ms_to_frames(self.fps,self.bout_duration_ms)        


@dataclass
class ConfigTrajSegmentation:
    """All parameters values relative to time should be in ms
    """
    fps: float
    peak_prominence: float = 0.4
    margin_before_peak_ms: float = 28
    bout_duration_ms: float = 200
    
    @property
    def margin_before_peak(self):
        return convert_ms_to_frames(self.fps,self.margin_before_peak_ms)        
    @property
    def bout_duration(self):
        return convert_ms_to_frames(self.fps,self.bout_duration_ms)        
 

@dataclass
class ConfigClassification:
    """All parameters values relative to time should be in ms
    """
    fps: float
    margin_before_peak_ms: float = 28
    bout_duration_ms: float = 200
    
    augment_min_delay_ms: float = -7
    augment_max_delay_ms: float = 20
    augment_step_delay_ms: float = 4
    
    feature_weight: np.ndarray = np.ones(10)
    N_kNN: int = 15
    
    dict_bouts: dict = field(init=False)

    
    def __post_init__(self):
        
        folder = Path(__file__).parent.parent
        filename = os.path.join(folder,"classification", "Bouts_dict.pickle")
        with open(filename, 'rb') as handle:
            self.bouts_dict = pickle.load(handle)
            
        #TODO NEED TO INCLUDE CHECK TO MAKE SURE THE DELAY ARE SUITED TO THE BOUTS DICTS
        
    
    @property
    def margin_before_peak(self):
        return convert_ms_to_frames(self.fps,self.margin_before_peak_ms)        
    @property
    def bout_duration(self):
        return convert_ms_to_frames(self.fps,self.bout_duration_ms)        
    @property
    def augment_min_delay(self):
        return convert_ms_to_frames(self.fps,self.augment_min_delay_ms)        
    @property
    def augment_max_delay(self):
        return convert_ms_to_frames(self.fps,self.augment_max_delay_ms)        
    @property
    def augment_step_delay(self):
        return convert_ms_to_frames(self.fps,self.augment_step_delay_ms)     




