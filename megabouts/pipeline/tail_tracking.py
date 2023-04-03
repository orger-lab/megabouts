import numpy as np
import pandas as pd
from dataclasses import dataclass,field
from functools import partial

from megabouts.pipeline.cfg import ConfigTailPreprocess,ConfigSparseCoding,ConfigTailSegmentation

from megabouts.tracking_data.dataset import Dataset_CentroidTracking,Dataset_TailTracking
from megabouts.preprocessing.preprocessing import preprocess_tail
from megabouts.sparse_coding.sparse_coding import compute_sparse_code,SparseCode
from megabouts.segmentation.segment import Segment,segment_from_code,segment_from_code_w_fine_alignement,extract_bouts
from megabouts.classification.classify import bouts_classifier,Classification
from megabouts.classification.template_bouts import Knn_Training_Dataset

@dataclass
class PipelineTailTracking_Result:
    tracking_data: Dataset_TailTracking
    tracking_data_clean: Dataset_TailTracking
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
        
        
    def preprocess(self,tail_angle):
        return preprocess_tail(tail_angle=tail_angle,
                               limit_na=self.cfg_preprocess.limit_na,
                               num_pcs=self.cfg_preprocess.num_pcs,
                               savgol_window = self.cfg_tail_preprocess.savgol_window,
                               baseline_method = self.cfg_preprocess.baseline_method,
                               baseline_params = self.cfg_preprocess.baseline_params)
        
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
        
        
    def run(self,tail_angle):
        
        # Preprocess Data:
        
        tracking_data = Dataset_TailTracking(fps=self.cfg_preprocess.fps,
                                             tail_angle=tail_angle)
        
        tail_angle_clean,baseline = self.preprocess(tail_angle=tracking_data.tail_angle)
        N_c = self.cfg_preprocess.tail_segment_cutoff
        tail_angle_detrend = tail_angle_clean[:,:N_c]-baseline[:,:N_c]
        
        tracking_data_clean = Dataset_TailTracking(fps=self.cfg_preprocess.fps,
                                                   tail_angle=tail_angle_clean-baseline)
        # Compute Sparse Code:
        sparse_code = self.compute_sparse_code(tail_angle_detrend)
        
        # Compute Segments:
        segments,segment_original,is_aligned = self.find_segment(z=sparse_code.z,tail_angle1d=tail_angle_detrend[:,N_c-1])
        
        tail_array = extract_bouts(tail_angle=tail_angle_detrend,
                                   segment = segments)
                
        # Compute result:
        res = PipelineTailTracking_Result(tracking_data=tracking_data,
                                          tracking_data_clean=tracking_data_clean,
                                          baseline = baseline,
                                          sparse_code = sparse_code,
                                          segments = segments,
                                          tail_array = tail_array)
        
        self.res = res
        
        return self.res