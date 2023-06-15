import numpy as np
import pandas as pd
from dataclasses import dataclass,field
from functools import partial

from megabouts.pipeline.cfg import ConfigTailPreprocess,ConfigSparseCoding,ConfigTailSegmentation

from megabouts.tracking_data.dataset import Dataset_CentroidTracking,Dataset_TailTracking
from megabouts.preprocessing.preprocessing import preprocess_tail,interp_tail_nan
from megabouts.sparse_coding.sparse_coding import compute_sparse_code,SparseCode
from megabouts.segmentation.segment import Segment,segment_from_code,segment_from_code_w_fine_alignement,extract_bouts
from megabouts.classification.classify import bouts_classifier,Classification
from megabouts.classification.template_bouts import Knn_Training_Dataset

@dataclass
class PipelineTailTracking_Result:
    tracking_data: Dataset_TailTracking
    tracking_data_clean: Dataset_TailTracking
    failed_tracking: np.ndarray=field(init=True,repr=False)
    baseline: np.ndarray=field(init=True,repr=False)
    sparse_code: SparseCode
    segments: Segment
    tail_array: np.ndarray=field(init=True,repr=False)
    
@dataclass(repr=False)
class PipelineTailTracking():

    cfg_preprocess : ConfigTailPreprocess = field(init=True)
    cfg_sparse_coding : ConfigSparseCoding = field(init=True)
    cfg_segment : ConfigTailSegmentation = field(init=True)
    knn_training_dataset: Knn_Training_Dataset = field(init=False)
    res: PipelineTailTracking_Result = field(init=False)

    def __post_init__(self):
        assert self.cfg_preprocess.fps==self.cfg_segment.fps==self.cfg_sparse_coding.fps, \
            f"fps should be the same in all configs"
        
    def preprocess_tail(self,tail_angle):
        # We don't use the last segments as they are noisy
        tail_angle_short = tail_angle[:,:self.cfg_tail_preprocess.num_tail_segments]
        smooth_tail_angle,baseline = preprocess_tail(tail_angle=tail_angle_short,
                                                    num_pcs=self.cfg_tail_preprocess.num_pcs,
                                                    savgol_window = self.cfg_tail_preprocess.savgol_window,
                                                    baseline_method = self.cfg_tail_preprocess.baseline_method,
                                                    baseline_params = self.cfg_tail_preprocess.baseline_params)
        tail_angle_detrend = smooth_tail_angle-baseline
        smooth_tail_speed = compute_tail_speed(tail_angle= tail_angle_detrend,
                                              fps=self.cfg_tail_preprocess.fps,
                                              tail_speed_filter=self.cfg_segment.tail_speed_filter,
                                              tail_speed_boxcar_filter=self.cfg_segment.tail_speed_boxcar_filter)
        return smooth_tail_angle,baseline,smooth_tail_speed
    
    def compute_sparse_code(self,tail_angle):
        return compute_sparse_code(tail_angle=tail_angle,
                                   Dict=self.cfg_sparse_coding.Dict,
                                   Wg=[],
                                   lmbda=self.cfg_sparse_coding.lmbda,
                                   gamma=self.cfg_sparse_coding.gamma,
                                   mu=self.cfg_sparse_coding.mu,
                                   Whn=self.cfg_sparse_coding.window_inhib)
    
    
    def find_segment(self,z,tail_angle1d):
        return segment_from_code_w_fine_alignement(z=z,tail_angle1d=tail_angle1d,
                                 min_code_height=self.cfg_segment.min_code_height,
                                 min_spike_dist=self.cfg_segment.min_spike_dist,
                                 bout_duration=self.cfg_segment.bout_duration,
                                 margin_before_peak=self.cfg_segment.margin_before_peak,
                                 dict_peak=self.cfg_sparse_coding.dict_peak)
        
    
    def run(self,**kwargs):
        
        if self.cfg_tail_preprocess.tail_input_type=='tail_angle':
            return self._run_from_tail_angle(tail_angle=kwargs['tail_angle'])
            
        elif self.cfg_tail_preprocess.tail_input_type=='keypoints':
            return self._run_from_keypoints(x = kwargs['x'],y=kwargs['y'],tail_x=kwargs['tail_x'],tail_y=kwargs['tail_y'])

        else:
            raise ValueError("tail preprocess tracking input should be 'tail_angle' or 'keypoints'")

    
    def _run_from_keypoints(self,body_x,body_y,tail_x,tail_y):
    
        T = tail_x.shape[0]
        N_keypoints = tail_x.shape[1]
        
        # Interpolate to correct number of segments:
        if N_keypoints!=11:
            tail_x,tail_y = interpolate_tail_keypoint(tail_x,tail_y,10)

        # Compute angle:
        tail_angle,body_angle = compute_angles_from_keypoints(body_x,body_y,tail_x,tail_y)
        return self._run_from_tail_angle(tail_angle)


    def _run_from_tail_angle(self,tail_angle):
        
        tracking_data = Dataset_TailTracking(fps=self.cfg_tail_preprocess.fps,
                                             tail_angle=tail_angle)
        
        # Make sure the input has the correct number of segments
        if tracking_data.tail_angle.shape[1]!=10:
            tail_angle = interpolate_tail_angle(tracking_data.tail_angle,n_segments=10)
        else:
            tail_angle = tracking_data.tail_angle

        tail_angle_no_nan, failed_tail_tracking,failed_tail_tracking_partial = interp_tail_nan(tail_angle,limit_na=self.cfg_tail_preprocess.limit_na)
        
        
        # Preprocess Data:
        smooth_tail_angle,baseline,smooth_tail_speed = self.preprocess_tail(tail_angle=tracking_data_input.tail_angle)
        tail_angle_detrend = smooth_tail_angle-baseline
        tracking_data_clean = Dataset_TailTracking(fps=self.cfg_tail_preprocess.fps,
                                                   tail_angle=tail_angle_detrend)

        # Compute Sparse Code:
        sparse_code = self.compute_sparse_code(tail_angle_detrend)
        
        # Compute Segments:
        segments,segment_original,is_aligned = self.find_segment(z=sparse_code.z,tail_angle1d=tail_angle_detrend)
        
        tail_array = extract_bouts(tail_angle=tail_angle_detrend,
                                   segment = segments)
                
        # Compute result:
        res = PipelineTailTracking_Result(tracking_data=tracking_data_input,
                                          tracking_data_clean=tracking_data_clean,
                                          failed_tracking = failed_tail_tracking,
                                          baseline = baseline,
                                          sparse_code = sparse_code,
                                          segments = segments,
                                          tail_array = tail_array)
        
        self.res = res
        
        return self.res