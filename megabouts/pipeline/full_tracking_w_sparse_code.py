import numpy as np
import pandas as pd
from dataclasses import dataclass,field
from functools import partial

from megabouts.pipeline.cfg import ConfigTrajPreprocess,ConfigTailPreprocess,ConfigSparseCoding,ConfigTailSegmentationFromSparseCode,ConfigClassification

from megabouts.tracking_data.dataset import Dataset_CentroidTracking,Dataset_TailTracking,Dataset_FullTracking
from megabouts.preprocessing.preprocessing import Preprocessed_Traj
from megabouts.segmentation.segment import Segment
from megabouts.classification.template_bouts import Knn_Training_Dataset

from megabouts.preprocessing.preprocessing import preprocess_traj,preprocess_tail
from megabouts.segmentation.segment import segment_from_kinematic_activity,extract_aligned_traj,extract_bouts

from megabouts.sparse_coding.sparse_coding import compute_sparse_code,SparseCode
from megabouts.segmentation.segment import Segment,segment_from_code,segment_from_code_w_fine_alignement
from megabouts.classification.classify import bouts_classifier,Classification

from megabouts.utils.utils_bouts import compute_bout_cat_ts


@dataclass
class PipelineFullTracking_Result:
    tracking_data: Dataset_FullTracking
    tracking_data_clean: Dataset_FullTracking
    baseline: np.ndarray=field(init=True,repr=False)
    sparse_code: SparseCode
    segments: Segment
    segments_original: Segment
    tail_and_traj_array: np.ndarray=field(init=True,repr=False)
    classification: Classification
    bout_category_ts: np.ndarray=field(init=True,repr=False)
    bout_category_ts_signed: np.ndarray=field(init=True,repr=False)
    
    
@dataclass(repr=False)
class PipelineFullTracking():
    
    cfg_tail_preprocess : ConfigTailPreprocess = field(init=True)
    cfg_traj_preprocess : ConfigTrajPreprocess = field(init=True)
    cfg_sparse_coding : ConfigSparseCoding = field(init=True)
    cfg_segment : ConfigTailSegmentationFromSparseCode = field(init=True)
    cfg_classify : ConfigClassification = field(init=True)
    knn_training_dataset_augmented: Knn_Training_Dataset = field(init=False)
    load_training: bool = True
    res: PipelineFullTracking_Result = field(init=False)
    
    def __post_init__(self):
        
        assert self.cfg_segment.bout_duration==self.cfg_classify.bout_duration, \
            f"bout duration should be the same in both config"
        
        assert self.cfg_tail_preprocess.fps==self.cfg_traj_preprocess.fps==self.cfg_segment.fps==self.cfg_classify.fps, \
            f"fps should be the same in both config"
        if self.load_training:
            self.load_training_template()
            
    def load_training_template(self):
        self.knn_training_dataset_augmented = Knn_Training_Dataset(fps = self.cfg_tail_preprocess.fps,
                                                                   augmentation_delays = np.unique(np.arange(self.cfg_classify.augment_min_delay,self.cfg_classify.augment_max_delay,self.cfg_classify.augment_step_delay).tolist()+[0]),
                                                                   bouts_dict = self.cfg_classify.bouts_dict,
                                                                   bout_duration =  self.cfg_classify.bout_duration,
                                                                   peak_loc = self.cfg_classify.margin_before_peak)
        
    def preprocess_traj(self,x,y,body_angle):
        return preprocess_traj(x=x,
                               y=y,
                               body_angle=body_angle,
                               fps=self.cfg_traj_preprocess.fps,
                               fc_min=self.cfg_traj_preprocess.freq_cutoff_min,
                               beta=self.cfg_traj_preprocess.beta,
                               robust_diff=self.cfg_traj_preprocess.robust_diff,
                               lag=self.cfg_traj_preprocess.lag_kinematic_activity)
    
    def preprocess_tail(self,tail_angle):
        return preprocess_tail(tail_angle=tail_angle,
                               limit_na=self.cfg_tail_preprocess.limit_na,
                               num_pcs=self.cfg_tail_preprocess.num_pcs,
                               baseline_method = self.cfg_tail_preprocess.baseline_method,
                               baseline_params = self.cfg_tail_preprocess.baseline_params)
        
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
    
    def classify(self,X):
        return bouts_classifier(X,
                                kNN_training_dataset=self.knn_training_dataset_augmented,
                                weight=self.cfg_classify.feature_weight,
                                n_neighbors=self.cfg_classify.N_kNN,
                                tracking_method='tail_and_traj')
        
    def run(self,tail_angle,x,y,body_angle):
        
        
        # Preprocess Data:
        tracking_data = Dataset_FullTracking(fps=self.cfg_tail_preprocess.fps,
                                             x=x,
                                             y=y,
                                             body_angle=body_angle,
                                             tail_angle=tail_angle)
        
        clean_traj = self.preprocess_traj(x=tracking_data.x,
                                          y=tracking_data.y,
                                          body_angle=tracking_data.body_angle)
        
        tail_angle_clean,baseline = self.preprocess_tail(tail_angle=tracking_data.tail_angle)
        N_c = self.cfg_tail_preprocess.tail_segment_cutoff
        tail_angle_detrend = tail_angle_clean[:,:N_c]-baseline[:,:N_c]
        
        tracking_data_clean = Dataset_FullTracking(fps=self.cfg_tail_preprocess.fps,
                                                   x=clean_traj.x,
                                                   y=clean_traj.y,
                                                   body_angle=clean_traj.body_angle,
                                                   tail_angle=tail_angle_clean-baseline)
        # Compute Sparse Code:
        sparse_code = self.compute_sparse_code(tail_angle_detrend)
        
        # Compute Segments:
        segments,segment_original,is_aligned = self.find_segment(z=sparse_code.z,tail_angle1d=tail_angle_detrend[:,N_c-1])
        
        traj_array = extract_aligned_traj(x = clean_traj.x,
                                          y = clean_traj.y,
                                          body_angle = clean_traj.body_angle,
                                          segment = segments)
        tail_array = extract_bouts(tail_angle=tail_angle_detrend,
                                   segment = segments)
        
        tail_and_traj_array = np.concatenate((tail_array,traj_array),axis=1)
        
        # Classify:
        classification_res = self.classify(tail_and_traj_array)
        
        # Refine segmentation:
        onset_shift = classification_res.onset_shift
        onset_refined = [on_ + int(onset_shift[i]) for i,on_ in enumerate(segments.onset)]
        offset_refined = [off_ + int(onset_shift[i]) for i,off_  in enumerate(segments.offset)]
    
        segments_refined = Segment(onset=onset_refined,offset=offset_refined,bout_duration=self.cfg_segment.bout_duration)
        traj_array = extract_aligned_traj(x = clean_traj.x,
                                          y = clean_traj.y,
                                          body_angle = clean_traj.body_angle,
                                          segment = segments_refined)
        tail_array = extract_bouts(tail_angle=tail_angle_detrend,
                                   segment = segments_refined)
        
        tail_and_traj_array_refined = np.concatenate((tail_array,traj_array),axis=1)
        
        # Compute Time series of categories:
        
        bout_category = classification_res.bout_category  
        bout_category_ts,bout_category_ts_signed = compute_bout_cat_ts(segments_refined.onset,segments_refined.offset,bout_category,tracking_data.n_frames)
        
        # Compute result:
        
        res = PipelineFullTracking_Result(tracking_data=tracking_data,
                                          tracking_data_clean = tracking_data_clean,
                                          baseline = baseline,
                                          sparse_code = sparse_code,
                                          segments = segments_refined,
                                          segments_original = segments,
                                          tail_and_traj_array = tail_and_traj_array_refined,
                                          classification = classification_res,
                                          bout_category_ts = bout_category_ts,
                                          bout_category_ts_signed = bout_category_ts_signed)

    
        self.res = res
        
        return self.res
    
    
    
    

