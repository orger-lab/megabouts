import numpy as np
import pandas as pd
from dataclasses import dataclass,field
from functools import partial

from pipeline.cfg import ConfigTrajPreprocess,ConfigTrajSegmentationClassification

from tracking_data.dataset import Dataset_CentroidTracking,Dataset_TailTracking
from preprocessing.preprocessing import Preprocessed_Traj
from segmentation.segment import Segment
from classification.template_bouts import Knn_Training_Dataset

from preprocessing.preprocessing import preprocess_traj
from segmentation.segment import segment_from_mobility,extract_aligned_traj
from classification.classify import bouts_classifier



# CENTROID TRACKING PIPELINE:


@dataclass(repr=False)
class PipelineCentroidTracking():

    cfg_preprocess : ConfigTrajPreprocess = field(init=True)
    cfg_segment_classify : ConfigTrajSegmentationClassification = field(init=True)
    knn_training_dataset: Knn_Training_Dataset = field(init=False)
    load_training: bool = True
    
    def __post_init__(self):
        assert self.cfg_preprocess.fps==self.cfg_segment_classify.fps, \
            f"fps should be the same in both config"
        if self.load_training:
            self.load_training_template()
            
    
    def load_training_template(self):
        self.knn_training_dataset_augmented = Knn_Training_Dataset(fps=self.cfg_preprocess.fps,
                                                         augmentation_delays=np.arange(0,
                                                                                       self.cfg_segment_classify.augment_max_delay,
                                                                                       self.cfg_segment_classify.augment_step_delay),
                                                         ignore_CS=True)
        self.knn_training_dataset = Knn_Training_Dataset(fps=self.cfg_preprocess.fps,
                                                         augmentation_delays=[0],
                                                         ignore_CS=True)
    def preprocess(self,x,y,body_angle):
        return preprocess_traj(x=x,
                               y=y,
                               body_angle=body_angle,
                               fps=self.cfg_preprocess.fps,
                               fc_min=self.cfg_preprocess.freq_cutoff_min,
                               beta=self.cfg_preprocess.beta,
                               robust_diff=self.cfg_preprocess.robust_diff,
                               lag=self.cfg_preprocess.lag_mobility)
        
    def find_segment(self,mobility):
        return segment_from_mobility(mobility=mobility,
                                     bout_duration=self.cfg_segment_classify.bout_duration,
                                     margin_before_peak=self.cfg_segment_classify.margin_before_peak,
                                     prominence=self.cfg_segment_classify.peak_prominence)
                                                 
    def align_classifier(self,X):
        return bouts_classifier(X,
                                kNN_training_dataset=self.knn_training_dataset_augmented,
                                weight=self.cfg_segment_classify.feature_weight,
                                n_neighbors=self.cfg_segment_classify.N_kNN,
                                tracking_method='traj')
        
    def classify(self,X):
        return bouts_classifier(X,
                                kNN_training_dataset=self.knn_training_dataset,
                                weight=self.cfg_segment_classify.feature_weight,
                                n_neighbors=self.cfg_segment_classify.N_kNN,
                                tracking_method='traj')
        
    def run(self,x,y,body_angle):
        
        tracking_data = Dataset_CentroidTracking(fps=self.cfg_preprocess.fps,
                                                 x=x,
                                                 y=y,
                                                 body_angle=body_angle)
        
        clean_traj = self.preprocess(x=tracking_data.x,
                                     y=tracking_data.y,
                                     body_angle=tracking_data.body_angle)
        
        segments = self.find_segment(mobility=clean_traj.mobility)
        
        traj_array = extract_aligned_traj(x = clean_traj.x,
                                          y = clean_traj.y,
                                          body_angle = clean_traj.body_angle,
                                          segment = segments)
        
        bout_category,onset_delay,id_nearest_template = self.align_classifier(traj_array)
        onset_refined = [on_ + int(onset_delay[i]) for i,on_ in enumerate(segments.onset)]
        offset_refined = [off_ + int(onset_delay[i]) for i,off_  in enumerate(segments.offset)]

        segment_refined = Segment(onset=onset_refined,offset=offset_refined,bout_duration=self.cfg_segment_classify.bout_duration)
        
        traj_array = extract_aligned_traj(x = clean_traj.x,
                                  y = clean_traj.y,
                                  body_angle = clean_traj.body_angle,
                                  segment = segment_refined)
        
        bout_category,_,id_nearest_template  = self.classify(traj_array)

        return tracking_data,clean_traj,segments,segment_refined,traj_array,bout_category,onset_delay,id_nearest_template