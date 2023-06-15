from dataclasses import dataclass,field

import numpy as np
from scipy.signal import find_peaks
from megabouts.segmentation.align import find_first_half_beat
from megabouts.segmentation.threshold import estimate_threshold_using_GMM
from megabouts.utils.utils import find_onset_offset_numpy

@dataclass(repr=False)
class Segment():
    onset: np.ndarray = field(init=True)
    offset: np.ndarray = field(init=True)
    bout_duration: int = field(init=True)
    
def segment_from_peaks(peaks,max_t,margin_before_peak=20,bout_duration=140):
    """_summary_

    Args:
        peaks (list): list of peak location. 
        max_t (int): duration of time series.
        margin_before_peaks (int): Defaults to 20
        bout_duration (int): Defaults to 140

    Returns:
        tuple(list,list): onset and offset of bouts
    """    
    onset = []
    offset = []
    for iter_,peak in enumerate(peaks):
        if ((peak>margin_before_peak)&(peak+bout_duration<max_t)):

            id_st = peak - margin_before_peak
            id_ed = id_st + bout_duration
            
            onset.append(id_st)
            offset.append(id_ed)

    return onset,offset
  
    
def extract_aligned_traj(x:np.ndarray,
                         y:np.ndarray,
                         body_angle:np.ndarray,
                         segment:type(Segment))->np.ndarray:
    """ Segment continuous trajectory into a tensor of segments
    the trajectory are aligned according to the initial position and angle

    Args:
        x (np.ndarray): x position in mm
        y (np.ndarray): y position in mm
        body_angle (np.ndarray()): yaw angle in radian

    Returns:
        np.ndarray: Tensor of concatenated x,y,body_angle segment
        of size (num_bouts,3,bout_duration)
    """        
    traj_array = np.zeros((len(segment.onset),3,segment.bout_duration))
    duration = segment.bout_duration
    for i,id_st in enumerate(segment.onset):
        id_ed = id_st + duration
        sub_x,sub_y,sub_body_angle = x[id_st:id_ed],y[id_st:id_ed],body_angle[id_st:id_ed]
        Pos = np.zeros((2,segment.bout_duration))
        Pos[0,:] = sub_x-sub_x[0]
        Pos[1,:] = sub_y-sub_y[0]
        theta=-sub_body_angle[0]
        body_angle_rotated=sub_body_angle-sub_body_angle[0]
        RotMat=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        PosRot=np.dot(RotMat,Pos)
        sub_x,sub_y,sub_body_angle = PosRot[0,:],PosRot[1,:],body_angle_rotated

        traj_array[i,0,:],traj_array[i,1,:],traj_array[i,2,:] = sub_x,sub_y,sub_body_angle
    
    return traj_array

    
def extract_bouts(tail_angle:np.ndarray,
                  segment:type(Segment))->np.ndarray:
    """ Segment continuous trajectory into a tensor of segments
    the trajectory are aligned according to the initial position and angle

    Args:
        tail_angle (np.ndarray): tail_angle of size (n_segments,n_frames)
    
    Returns:
        np.ndarray: Tensor of concatenated tail_angle
        of size (num_bouts,n_segments,bout_duration)
    """
    tail_array = np.zeros((len(segment.onset),tail_angle.shape[1],segment.bout_duration))
    duration = segment.bout_duration
    for i,id_st in enumerate(segment.onset):
        id_ed = id_st + duration
        tail_array[i,:,:]=tail_angle[id_st:id_ed,:].T
        
    return tail_array


def segment_from_kinematic_activity(*,kinematic_activity,bout_duration,margin_before_peak,prominence=0.4):
    """_summary_

    Args:
        kinematic_activity (np.ndarry): kinematic activity whose peak signal onset of bouts.
        bout_duration (int, optional): Distance between peaks in frames
        margin_before_peak (int, optional): onset will be peak location shifted by the margin,
        prominence (float, optional): prominence of peaks in kinematic activity. Defaults to 0.4.

    Returns:
        Segment: contains onset offset and duration of bouts
    """    
    peaks, _ = find_peaks(kinematic_activity,distance=bout_duration,prominence=prominence)
    peaks_bin = np.zeros(kinematic_activity.shape[0])
    peaks_bin[peaks]=1
    onset_init,offset_init = segment_from_peaks(peaks=peaks,max_t=len(kinematic_activity),margin_before_peak=margin_before_peak,bout_duration=bout_duration)
    segment = Segment(onset=onset_init,offset=offset_init,bout_duration=bout_duration)
        
    return segment
    
    
def segment_from_tail_speed(*,tail_angle,smooth_tail_speed,missed_frame,
                            tail_speed_thresh_std=2.1,
                            tail_speed_thresh_default=10,
                            min_bout_duration=80,
                            bout_duration=140,
                            margin_before_peak=20,
                            half_BC_filt = 150,
                            std_thresh = 5,
                            min_size_blob = 500):
    
    # Compute Threshold:
    if len(smooth_tail_speed[missed_frame==False])>10000:
        Thresh,ax = estimate_threshold_using_GMM(smooth_tail_speed[missed_frame==False],margin_std=2.1,axis=None)
    else:
        Thesh = tail_speed_thresh_default
        print('tail_speed_thresh_default is being used since recording is to short for estimation')
        
    tail_active = smooth_tail_speed>Thresh

    # Remove bouts of short duration
    onset,offset,duration = find_onset_offset_numpy(tail_active)
    onset_init,offset_init,duration = onset[duration>min_bout_duration],offset[duration>min_bout_duration],duration[duration>min_bout_duration]
    tail_active = 0*tail_active
    for on_,off_ in zip(onset_init,offset_init):
        tail_active[on_:off_]=1

    onset_aligned,offset_aligned = [],[]
    onset_original,offset_original = [],[]
    is_aligned = []
    
    for iter_,(on_,off_) in enumerate(zip(onset_init,offset_init)):
            if (on_+bout_duration<tail_angle.shape[0]):
                
                tail_bout = tail_angle[on_:off_,:]
                
                onset_original.append(int(on_))
                offset_original.append(int(off_))
                
                peak_location = find_first_half_beat(tail_bout,half_BC_filt = half_BC_filt, std_thresh = std_thresh,min_size_blob = min_size_blob)
                if np.isnan(peak_location):
                    peak_location = margin_before_peak
                    is_aligned.append(0)
                else:
                    is_aligned.append(1)

                id_st = np.round(on_ + peak_location-margin_before_peak)
                id_ed = np.round(off_)
                if (id_st+bout_duration<tail_angle.shape[0]):
                    onset_aligned.append(int(id_st))
                    offset_aligned.append(int(id_ed))
    
    # Remove bouts with NaN:
    onset_no_nan,offset_no_nan = [],[]
    for on_,off_ in zip(onset_original,offset_original):
        if np.any(missed_frame[on_:off_])==False:
            onset_no_nan.append(on_)
            offset_no_nan.append(off_)
    
    onset_aligned_no_nan,offset_aligned_no_nan = [],[]
    for on_,off_ in zip(onset_aligned,offset_aligned):
        if np.any(missed_frame[on_:off_])==False:
            onset_aligned_no_nan.append(on_)
            offset_aligned_no_nan.append(off_)
      
    segment_original = Segment(onset=onset_original,offset=offset_aligned,bout_duration=bout_duration)
    segment = Segment(onset=onset_aligned,offset=offset_aligned,bout_duration=bout_duration)
    
    return segment,segment_original,np.array(is_aligned),Thresh


def segment_from_code_w_fine_alignement(*,z,tail_angle1d,
                                        min_code_height=1,
                                        min_spike_dist=120,
                                        bout_duration=140,
                                        margin_before_peak=20,
                                        dict_peak=20,
                                        half_BC_filt = 150,
                                        std_thresh = 5,
                                        min_size_blob = 500):
    
    # FINDING PEAKS IN SPARSE CODE:
    z_max = np.max(np.abs(z),axis=1)
    peaks, _ = find_peaks(z_max, height=min_code_height,distance=min_spike_dist)
    onset,offset = [],[]
    is_aligned = []
    onset_original,offset_original = [],[]
    
    for iter_,peak in enumerate(peaks):
            if ((peak>margin_before_peak)&(peak+bout_duration<tail_angle1d.shape[0]-margin_before_peak)):
                id_st = peak
                id_ed = id_st + bout_duration
                tail_bout = tail_angle1d[id_st:id_ed]
                
                onset_original.append(int(id_st-dict_peak))
                offset_original.append(int(id_st-dict_peak+bout_duration))
                
                peak_location = find_first_half_beat(tail_bout,half_BC_filt = half_BC_filt, std_thresh = std_thresh,min_size_blob = min_size_blob)
                if np.isnan(peak_location):
                    peak_location = dict_peak
                    is_aligned.append(0)
                else:
                    is_aligned.append(1)

                id_st = np.round(id_st + peak_location-margin_before_peak)
                id_ed = np.round(id_st + bout_duration)
                onset.append(int(id_st))
                offset.append(int(id_ed))
            
                
                
    segment_original = Segment(onset=onset_original,offset=offset_original,bout_duration=bout_duration)
    segment = Segment(onset=onset,offset=offset,bout_duration=bout_duration)
    
    return segment,segment_original,np.array(is_aligned)

def segment_from_code(*,z,
                      min_code_height=1,
                      min_spike_dist=120,
                      bout_duration=140,
                      margin_before_peak=20):

    # FINDING PEAKS IN SPARSE CODE:
    z_max = np.max(np.abs(z),axis=1)
    peaks, _ = find_peaks(z_max, height=min_code_height,distance=min_spike_dist)
    peaks_bin = np.zeros(z.shape[0])
    peaks_bin[peaks]=1
    onset_init,offset_init = segment_from_peaks(peaks=peaks,max_t=len(z_max),margin_before_peak=margin_before_peak,bout_duration=bout_duration)
    segment = Segment(onset=onset_init,offset=offset_init,bout_duration=bout_duration)
    return segment