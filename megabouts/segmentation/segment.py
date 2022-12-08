from dataclasses import dataclass,field

import numpy as np
from scipy.signal import find_peaks
from megabouts.segmentation.align import align_bout_peaks

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
    for i,(id_st,id_ed) in enumerate(zip(segment.onset,segment.offset)):
        
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
    for i,(id_st,id_ed) in enumerate(zip(segment.onset,segment.offset)):

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

def segment_from_code_w_fine_alignement(*,z,tail_angle1d,
                                        min_code_height=1,
                                        min_spike_dist=120,
                                        bout_duration=140,
                                        margin_before_peak=20,
                                        dict_peak=20):
    
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
                
                try:
                    peak_location = align_bout_peaks(tail_bout,quantile_threshold = 0.25 , minimum_peak_size = 0.25, minimum_peak_to_peak_amplitude = 4,debug_plot_axes=None)
                except:
                    peak_location = np.nan
                    
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
 