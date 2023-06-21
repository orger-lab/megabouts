import numpy as np
import pandas as pd
from dataclasses import dataclass,field
from functools import partial

from megabouts.pipeline.cfg import ConfigTrajPreprocess,ConfigTailPreprocess,ConfigHalfBeat,ConfigTailSegmentation,ConfigClassification

from megabouts.tracking_data.dataset import Dataset_CentroidTracking,Dataset_TailTracking,Dataset_FullTracking
from megabouts.tracking_data.convert_tracking import compute_angles_from_keypoints,interpolate_tail_keypoint,interpolate_tail_angle
from megabouts.preprocessing.preprocessing import Preprocessed_Traj
from megabouts.segmentation.segment import Segment
from megabouts.classification.template_bouts import Knn_Training_Dataset

from megabouts.preprocessing.preprocessing import interp_tail_nan,interp_traj_nan
from megabouts.preprocessing.preprocessing import preprocess_traj,preprocess_tail,compute_tail_speed
from megabouts.segmentation.segment import segment_from_kinematic_activity,extract_aligned_traj,extract_bouts

from megabouts.sparse_coding.sparse_coding import compute_sparse_code,SparseCode
from megabouts.segmentation.segment import Segment,segment_from_tail_speed
from megabouts.classification.classify import bouts_classifier,Classification

from megabouts.utils.utils_bouts import compute_bout_cat_ts


@dataclass
class PipelineFullTracking_Result:
    tracking_data: Dataset_FullTracking
    tracking_data_clean: Dataset_FullTracking
    failed_tracking: np.ndarray=field(init=True,repr=False)
    baseline: np.ndarray=field(init=True,repr=False)
    smooth_tail_speed: np.ndarray=field(init=True,repr=False)
    tail_speed_thresh : float
    segments: Segment
    tail_and_traj_array: np.ndarray=field(init=True,repr=False)
    bout_category: np.ndarray=field(init=True,repr=False)
    proba: np.ndarray=field(init=True,repr=False)
    outlier_score: np.ndarray=field(init=True,repr=False)
    id_nearest_template: np.ndarray=field(init=True,repr=False)
    id_nearest_template_aligned: np.ndarray=field(init=True,repr=False)
    bout_category_ts: np.ndarray=field(init=True,repr=False)

    
@dataclass(repr=False)
class PipelineFullTracking():
    
    cfg_tail_preprocess : ConfigTailPreprocess = field(init=True)
    cfg_traj_preprocess : ConfigTrajPreprocess = field(init=True)
    cfg_half_beat : ConfigHalfBeat = field(init=True)
    cfg_segment : ConfigTailSegmentation = field(init=True)
    cfg_classify : ConfigClassification = field(init=True)
    knn_training_dataset_augmented: Knn_Training_Dataset = field(init=False)
    load_training: bool = True
    res: PipelineFullTracking_Result = field(init=False)
    
    def __post_init__(self):
        
        assert self.cfg_segment.bout_duration==self.cfg_classify.bout_duration, \
            f"bout duration should be the same in both config"
        
        assert self.cfg_tail_preprocess.fps==self.cfg_traj_preprocess.fps==self.cfg_half_beat.fps==self.cfg_segment.fps==self.cfg_classify.fps, \
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


    def find_segment(self,tail_angle,smooth_tail_speed,missed_frame):
        return segment_from_tail_speed(tail_angle=tail_angle,
                                       smooth_tail_speed=smooth_tail_speed,
                                       missed_frame=missed_frame,
                                       tail_speed_thresh_std=self.cfg_segment.tail_speed_thresh_std,
                                       segment_peak_loc = self.cfg_segment.segment_peak_loc,
                                       min_bout_duration=self.cfg_segment.min_bout_duration,
                                       bout_duration=self.cfg_segment.bout_duration,
                                       margin_before_peak=self.cfg_segment.margin_before_peak,
                                       half_BC_filt = self.cfg_half_beat.half_BC_filt,
                                       std_thresh = self.cfg_half_beat.std_thresh,
                                       min_size_blob = self.cfg_half_beat.min_size_blob)
    
    def classify(self,X):
        return bouts_classifier(X,
                                kNN_training_dataset=self.knn_training_dataset_augmented,
                                weight=self.cfg_classify.feature_weight,
                                n_neighbors=self.cfg_classify.N_kNN,
                                tracking_method='tail_and_traj')
        
    
    def run(self,**kwargs):
        
        if self.cfg_tail_preprocess.tail_input_type=='tail_angle':
            return self._run_from_tail_angle(x = kwargs['x'],y=kwargs['y'],body_angle=kwargs['body_angle'],tail_angle=kwargs['tail_angle'])
            
        elif self.cfg_tail_preprocess.tail_input_type=='keypoints':
            return self._run_from_keypoints(body_x = kwargs['x'],body_y=kwargs['y'],tail_x=kwargs['tail_x'],tail_y=kwargs['tail_y'])

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
        return self._run_from_tail_angle(body_x,body_y,body_angle,tail_angle)


    def _run_from_tail_angle(self,x,y,body_angle,tail_angle):
        
        tracking_data = Dataset_TailTracking(fps=self.cfg_tail_preprocess.fps,
                                            tail_angle=tail_angle)
        
        # Make sure the input has the correct number of segments
        if tracking_data.tail_angle.shape[1]!=10:
            tail_angle = interpolate_tail_angle(tracking_data.tail_angle,n_segments=10)
        else:
            tail_angle = tracking_data.tail_angle

        x,y,body_angle,failed_traj_tracking = interp_traj_nan(x,y,body_angle,limit_na=self.cfg_traj_preprocess.limit_na)
        tail_angle_no_nan, failed_tail_tracking,failed_tail_tracking_partial = interp_tail_nan(tail_angle,limit_na=self.cfg_tail_preprocess.limit_na)
        
        failed_tracking = np.logical_or(failed_tail_tracking,failed_traj_tracking)
        tracking_data_input = Dataset_FullTracking(fps=self.cfg_tail_preprocess.fps,
                                            x=x,
                                            y=y,
                                            body_angle=body_angle,
                                            tail_angle=tail_angle_no_nan)
        
        # Preprocess Data:
        print('Preprocessing Trajectory')
        clean_traj = self.preprocess_traj(x=tracking_data_input.x,
                                            y=tracking_data_input.y,
                                            body_angle=tracking_data_input.body_angle)
        
        print('Preprocessing Tail')
        smooth_tail_angle,baseline,smooth_tail_speed = self.preprocess_tail(tail_angle=tracking_data_input.tail_angle)
        tail_angle_detrend = smooth_tail_angle-baseline
        tracking_data_clean = Dataset_FullTracking(fps=self.cfg_tail_preprocess.fps,
                                                    x=clean_traj.x,
                                                    y=clean_traj.y,
                                                    body_angle=clean_traj.body_angle,
                                                    tail_angle=tail_angle_detrend)

        # Compute Segments:
        print('Segmentation')
        segments,segment_original,is_aligned,Thresh = self.find_segment(tail_angle=tail_angle_detrend,
                                                                        smooth_tail_speed=smooth_tail_speed,
                                                                        missed_frame=failed_tracking)#tracking_data.missed_frame)
        
        traj_array = extract_aligned_traj(x = clean_traj.x,
                                        y = clean_traj.y,
                                        body_angle = clean_traj.body_angle,
                                        segment = segments)
        tail_array = extract_bouts(tail_angle=tail_angle_detrend,
                                segment = segments)
        
        tail_and_traj_array = np.concatenate((tail_array,traj_array),axis=1)
        
        # Classify:
        print('Classification')
        classification_res = self.classify(tail_and_traj_array)
        outlier_score = classification_res.outlier_score
        proba = classification_res.proba
        id_nearest_template = classification_res.id_nearest_template
        id_nearest_template_aligned = classification_res.id_nearest_template_aligned
        onset_shift = classification_res.onset_shift
        
        # Segmentation, technique: 'original' vs 'refined'
        if self.cfg_classify.refine_segmentation:
            
            # Refine segmentation:
            onset_refined = [on_ + int(onset_shift[i]) for i,on_ in enumerate(segments.onset)]
            offset_refined = segments.offset
            segments_refined = Segment(onset=onset_refined,offset=offset_refined,bout_duration=self.cfg_segment.bout_duration)
            traj_array = extract_aligned_traj(x = clean_traj.x,
                                                y = clean_traj.y,
                                                body_angle = clean_traj.body_angle,
                                                segment = segments_refined)
            tail_array = extract_bouts(tail_angle=tail_angle_detrend,
                                        segment = segments_refined)
            tail_and_traj_array = np.concatenate((tail_array,traj_array),axis=1)
            bout_category = classification_res.bout_category  
            

        else:
            segments_refined = segments
            bout_category = classification_res.bout_category  
            bout_sign = np.array([np.sign(tail_array[i,-1,self.cfg_segment.margin_before_peak]) for i in range(tail_array.shape[0])])
            bout_category = np.array([b if sg==1 else b+11 for b,sg in zip(bout_category%11,bout_sign)])


        bout_category_ts,bout_category_ts_signed = compute_bout_cat_ts(segments_refined.onset,segments_refined.offset,bout_category,tracking_data.n_frames)


        # Compute result:
        res = PipelineFullTracking_Result(tracking_data=tracking_data_input,
                                          tracking_data_clean = tracking_data_clean,
                                          failed_tracking = failed_tracking,
                                          baseline = baseline,
                                          smooth_tail_speed = smooth_tail_speed,
                                          tail_speed_thresh = Thresh,
                                          segments = segments_refined,
                                          tail_and_traj_array = tail_and_traj_array,
                                          bout_category = bout_category,
                                          proba = proba,
                                          outlier_score = outlier_score,
                                          id_nearest_template = id_nearest_template,
                                          id_nearest_template_aligned = id_nearest_template_aligned,
                                          bout_category_ts = bout_category_ts
                                          )

        return res
