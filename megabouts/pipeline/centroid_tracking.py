import numpy as np
import pandas as pd
from dataclasses import dataclass,field
from functools import partial

from megabouts.pipeline.cfg import ConfigTrajPreprocess,ConfigTrajSegmentation,ConfigClassification

from megabouts.tracking_data.dataset import Dataset_CentroidTracking
from megabouts.preprocessing.preprocessing import Preprocessed_Traj
from megabouts.segmentation.segment import Segment
from megabouts.classification.template_bouts import Knn_Training_Dataset

from megabouts.preprocessing.preprocessing import preprocess_traj
from megabouts.segmentation.segment import segment_from_kinematic_activity,extract_aligned_traj
from megabouts.classification.classify import bouts_classifier,Classification

from megabouts.utils.utils_bouts import compute_bout_cat_ts


@dataclass
class PipelineCentroidTracking_Result:
    tracking_data: Dataset_CentroidTracking
    tracking_data_clean: Dataset_CentroidTracking
    kinematic_activity: np.ndarray=field(init=True,repr=False)
    segments: Segment
    segments_original: Segment
    traj_array: np.ndarray=field(init=True,repr=False)
    classification: Classification
    bout_category_ts: np.ndarray=field(init=True,repr=False)
    bout_category_ts_signed: np.ndarray=field(init=True,repr=False)
    
    
@dataclass(repr=False)
class PipelineCentroidTracking():

    cfg_preprocess : ConfigTrajPreprocess = field(init=True)
    cfg_segment : ConfigTrajSegmentation = field(init=True)
    cfg_classify : ConfigClassification = field(init=True)
    knn_training_dataset: Knn_Training_Dataset = field(init=False)
    knn_training_dataset_augmented: Knn_Training_Dataset = field(init=False)
    load_training: bool = True
    res: PipelineCentroidTracking_Result = field(init=False)

    def __post_init__(self):
        
        assert self.cfg_segment.bout_duration==self.cfg_classify.bout_duration, \
            f"bout duration should be the same in both config"
        
        assert self.cfg_preprocess.fps==self.cfg_segment.fps==self.cfg_classify.fps, \
            f"fps should be the same in both config"
        if self.load_training:
            self.load_training_template()
            
    
    def load_training_template(self):
        self.knn_training_dataset_augmented = Knn_Training_Dataset(fps = self.cfg_preprocess.fps,
                                                                   augmentation_delays = np.unique(np.arange(self.cfg_classify.augment_min_delay,self.cfg_classify.augment_max_delay,self.cfg_classify.augment_step_delay).tolist()+[0]),
                                                                   bouts_dict = self.cfg_classify.bouts_dict,
                                                                   bout_duration =  self.cfg_classify.bout_duration,
                                                                   peak_loc = self.cfg_classify.margin_before_peak)
        
        self.knn_training_dataset = Knn_Training_Dataset(fps = self.cfg_preprocess.fps,
                                                                   augmentation_delays =[0],
                                                                   bouts_dict = self.cfg_classify.bouts_dict,
                                                                   bout_duration =  self.cfg_classify.bout_duration,
                                                                   peak_loc = self.cfg_classify.margin_before_peak)

    def preprocess(self,x,y,body_angle):
        return preprocess_traj(x=x,
                               y=y,
                               body_angle=body_angle,
                               fps=self.cfg_preprocess.fps,
                               fc_min=self.cfg_preprocess.freq_cutoff_min,
                               beta=self.cfg_preprocess.beta,
                               robust_diff=self.cfg_preprocess.robust_diff,
                               lag=self.cfg_preprocess.lag_kinematic_activity)
        
    def find_segment(self,kinematic_activity):
        return segment_from_kinematic_activity(kinematic_activity=kinematic_activity,
                                               bout_duration=self.cfg_segment.bout_duration,
                                               margin_before_peak=self.cfg_segment.margin_before_peak,
                                               prominence=self.cfg_segment.peak_prominence)
                                                 
    def align_classifier(self,X):
        return bouts_classifier(X,
                                kNN_training_dataset=self.knn_training_dataset_augmented,
                                weight=self.cfg_classify.feature_weight,
                                n_neighbors=self.cfg_classify.N_kNN,
                                tracking_method='traj')
        
    def classify(self,X):
        return bouts_classifier(X,
                                kNN_training_dataset=self.knn_training_dataset,
                                weight=self.cfg_classify.feature_weight,
                                n_neighbors=self.cfg_classify.N_kNN,
                                tracking_method='traj')
        
    def run(self,x,y,body_angle):
        
        # Preprocess Data:
        tracking_data = Dataset_CentroidTracking(fps=self.cfg_preprocess.fps,
                                                 x=x,
                                                 y=y,
                                                 body_angle=body_angle)
        
        clean_traj = self.preprocess(x=tracking_data.x,
                                     y=tracking_data.y,
                                     body_angle=tracking_data.body_angle)
        
        tracking_data_clean = Dataset_CentroidTracking(fps=self.cfg_preprocess.fps,
                                                       x=clean_traj.x,
                                                       y=clean_traj.y,
                                                       body_angle=clean_traj.body_angle)
        
        # Segment:
        segments = self.find_segment(kinematic_activity=clean_traj.kinematic_activity)
        
        traj_array = extract_aligned_traj(x = clean_traj.x,
                                          y = clean_traj.y,
                                          body_angle = clean_traj.body_angle,
                                          segment = segments)
        # Classify Align:
                
        classification_res = self.align_classifier(traj_array)

        # Refine segmentation:
        onset_shift = classification_res.onset_shift
        onset_refined = [on_ + int(onset_shift[i]) for i,on_ in enumerate(segments.onset)]
        offset_refined = [off_ + int(onset_shift[i]) for i,off_  in enumerate(segments.offset)]
        segment_refined = Segment(onset=onset_refined,offset=offset_refined,bout_duration=self.cfg_segment.bout_duration)
        traj_array = extract_aligned_traj(x = clean_traj.x,
                                  y = clean_traj.y,
                                  body_angle = clean_traj.body_angle,
                                  segment = segment_refined)
        
        # Classify :
        classification_res  = self.classify(traj_array)


        # Compute Time series of categories:
        
        bout_category = classification_res.bout_category  
        bout_category_ts,bout_category_ts_signed = compute_bout_cat_ts(segment_refined.onset,segment_refined.offset,bout_category,tracking_data.n_frames)
        
        # Compute result:
        
        res = PipelineCentroidTracking_Result(tracking_data=tracking_data,
                                              tracking_data_clean = tracking_data_clean,
                                              kinematic_activity = clean_traj.kinematic_activity,
                                              segments = segment_refined,
                                              segments_original = segments,
                                              traj_array = traj_array,
                                              classification = classification_res,
                                              bout_category_ts = bout_category_ts,
                                              bout_category_ts_signed = bout_category_ts_signed)
        self.res = res
        
        return self.res