import numpy as np
import pandas as pd
from dataclasses import dataclass,field
from functools import partial

from pipeline.cfg import ConfigTrajPreprocess,ConfigTailPreprocess,ConfigSparseCoding,ConfigTailSegmentationClassification

from tracking_data.dataset import Dataset_CentroidTracking,Dataset_TailTracking,Dataset_FullTracking
from preprocessing.preprocessing import Preprocessed_Traj
from segmentation.segment import Segment
from classification.template_bouts import Knn_Training_Dataset

from preprocessing.preprocessing import preprocess_traj,preprocess_tail
from segmentation.segment import segment_from_mobility,extract_aligned_traj,extract_bouts

from sparse_coding.sparse_coding import compute_sparse_code
from segmentation.segment import Segment,segment_from_code,segment_from_code_w_fine_alignement
from classification.classify import bouts_classifier


# FULL TRACKING PIPELINE:


@dataclass(repr=False)
class PipelineFullTracking():
    
    cfg_tail_preprocess : ConfigTailPreprocess = field(init=True)
    cfg_traj_preprocess : ConfigTrajPreprocess = field(init=True)

    cfg_sparse_coding : ConfigSparseCoding = field(init=True)

    cfg_segment_classify : ConfigTailSegmentationClassification = field(init=True)
    knn_training_dataset_augmented: Knn_Training_Dataset = field(init=False)
    load_training: bool = True
    
    
    def __post_init__(self):
        assert self.cfg_tail_preprocess.fps==self.cfg_traj_preprocess.fps==self.cfg_segment_classify.fps, \
            f"fps should be the same in both config"
        if self.load_training:
            self.load_training_template()
            
    def load_training_template(self):
        self.knn_training_dataset_augmented = Knn_Training_Dataset(fps=self.cfg_tail_preprocess.fps,
                                                         augmentation_delays=np.arange(0,
                                                                                       self.cfg_segment_classify.augment_max_delay,
                                                                                       self.cfg_segment_classify.augment_step_delay),
                                                         ignore_CS=True)
        
    def preprocess_traj(self,x,y,body_angle):
        return preprocess_traj(x=x,
                               y=y,
                               body_angle=body_angle,
                               fps=self.cfg_traj_preprocess.fps,
                               fc_min=self.cfg_traj_preprocess.freq_cutoff_min,
                               beta=self.cfg_traj_preprocess.beta,
                               robust_diff=self.cfg_traj_preprocess.robust_diff,
                               lag=self.cfg_traj_preprocess.lag_mobility)
    
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
                                 min_code_height=self.cfg_segment_classify.min_code_height,
                                 min_spike_dist=self.cfg_segment_classify.min_spike_dist,
                                 bout_duration=self.cfg_segment_classify.bout_duration,
                                 margin_before_peak=self.cfg_segment_classify.margin_before_peak)
    
    def classify(self,X):
        return bouts_classifier(X,
                                kNN_training_dataset=self.knn_training_dataset_augmented,
                                weight=self.cfg_segment_classify.feature_weight,
                                n_neighbors=self.cfg_segment_classify.N_kNN,
                                tracking_method='tail_and_traj')
        
    def run(self,tail_angle,x,y,body_angle):
        
        tracking_data = Dataset_FullTracking(fps=self.cfg_tail_preprocess.fps,
                                             x=x,
                                             y=y,
                                             body_angle=body_angle,
                                             tail_angle=tail_angle)
        
        clean_traj = self.preprocess_traj(x=tracking_data.x,
                                          y=tracking_data.y,
                                          body_angle=tracking_data.body_angle)
        
        tail_angle_clean,baseline = self.preprocess_tail(tail_angle=tracking_data.tail_angle)
        tail_angle_detrend = tail_angle_clean[:,:7]-baseline[:,:7]
        z,tail_angle_hat,decomposition = self.compute_sparse_code(tail_angle_detrend)
        segments = self.find_segment(z=z,tail_angle1d=tail_angle_detrend[:,6])
        
        traj_array = extract_aligned_traj(x = clean_traj.x,
                                          y = clean_traj.y,
                                          body_angle = clean_traj.body_angle,
                                          segment = segments)
        tail_array = extract_bouts(tail_angle=tail_angle_detrend,
                                   segment = segments)
        
        tail_and_traj_array = np.concatenate((tail_array,traj_array),axis=1)
        
        
        bout_category,onset_delay,id_nearest_template = self.classify(tail_and_traj_array)
        onset_refined = [on_ + int(onset_delay[i]) for i,on_ in enumerate(segments.onset)]
        offset_refined = [off_ + int(onset_delay[i]) for i,off_  in enumerate(segments.offset)]

        i = id_nearest_template
        N = len(np.where(self.knn_training_dataset_augmented.delays==0)[0])/2
        N_mid = len(self.knn_training_dataset_augmented.delays)/2
        sg = [-1 if i>N_mid else 1 for i in id_nearest_template]
        mod_,div_ = np.divmod(id_nearest_template, N)

        id_nearest_template = np.array([int(i) if s>0 else int(i+N_mid) for i,s in zip(div_,sg)])


        segments_refined = Segment(onset=onset_refined,offset=offset_refined,bout_duration=self.cfg_segment_classify.bout_duration)

        traj_array = extract_aligned_traj(x = clean_traj.x,
                                          y = clean_traj.y,
                                          body_angle = clean_traj.body_angle,
                                          segment = segments_refined)
        tail_array = extract_bouts(tail_angle=tail_angle_detrend,
                                   segment = segments_refined)
        
        tail_and_traj_array = np.concatenate((tail_array,traj_array),axis=1)
        

        return tracking_data,clean_traj,baseline,tail_angle_detrend,z,segments_refined,tail_and_traj_array,bout_category,id_nearest_template