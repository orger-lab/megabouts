import numpy as np
from scipy.signal import find_peaks
from segmentation.align import align_bout_peaks




def segment_from_peaks(peaks,T,Margin_before_peak=20,Bout_Duration=140):

    onset = []
    offset = []
    for iter_,peak in enumerate(peaks):
        if ((peak>Margin_before_peak)&(peak+Bout_Duration<T)):

            id_st = peak - Margin_before_peak
            id_ed = id_st + Bout_Duration
            
            onset.append(id_st)
            offset.append(id_ed)

    return onset,offset

def collect_bouts_traj(traj,onset,offset,Bout_Duration):
    
    bouts_array = np.zeros((len(onset),Bout_Duration,3))
    bouts_array_flat = np.zeros((len(offset),Bout_Duration*3))

    for i,(id_st,id_ed) in enumerate(zip(onset,offset)):

        bouts_array[i,:,:] = traj[id_st:id_ed,:]
        sub_x,sub_y,sub_body_angle = bouts_array[i,:,0],bouts_array[i,:,1],bouts_array[i,:,2]
        Pos = np.zeros((2,Bout_Duration))
        Pos[0,:] = sub_x-sub_x[0]
        Pos[1,:] = sub_y-sub_y[0]
        theta=-sub_body_angle[0]

        body_angle_rotated=sub_body_angle-sub_body_angle[0]
        RotMat=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        PosRot=np.dot(RotMat,Pos)
        sub_x,sub_y,sub_body_angle = PosRot[0,:],PosRot[1,:],body_angle_rotated

        bouts_array[i,:,0],bouts_array[i,:,1],bouts_array[i,:,2] = sub_x,sub_y,sub_body_angle
        bouts_array_flat[i,:Bout_Duration],bouts_array_flat[i,Bout_Duration:Bout_Duration*2],bouts_array_flat[i,Bout_Duration*2:] = sub_x, sub_y, sub_body_angle

    return bouts_array,bouts_array_flat


def create_segmentation_from_mobility(Min_Ampl=1.4,Bout_Duration=140,Margin_before_peak = 20):

    def segment_from_mobility(mobility,traj):

        peaks, _ = find_peaks(mobility, height=1.4,distance=Bout_Duration)

        bouts_array = np.zeros((len(peaks),Bout_Duration,3))
        bouts_array_flat = np.zeros((len(peaks),Bout_Duration*3))

        onset = []
        offset = []
        i = 0
        for iter_,peak in enumerate(peaks):
                if ((peak>Margin_before_peak)&(peak+Bout_Duration<mobility.shape[0])):
                    id_st = peak - Margin_before_peak
                    id_ed = id_st + Bout_Duration
                    bouts_array[i,:,:] = traj[id_st:id_ed,:]

                    sub_x,sub_y,sub_body_angle = bouts_array[i,:,0],bouts_array[i,:,1],bouts_array[i,:,2]
                    Pos = np.zeros((2,Bout_Duration))
                    Pos[0,:] = sub_x-sub_x[0]
                    Pos[1,:] = sub_y-sub_y[0]
                    theta=-sub_body_angle[0]

                    body_angle_rotated=sub_body_angle-sub_body_angle[0]
                    RotMat=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                    PosRot=np.dot(RotMat,Pos)
                    sub_x,sub_y,sub_body_angle = PosRot[0,:],PosRot[1,:],body_angle_rotated

                    bouts_array[i,:,0],bouts_array[i,:,1],bouts_array[i,:,2] = sub_x,sub_y,sub_body_angle
                    bouts_array_flat[i,:Bout_Duration],bouts_array_flat[i,Bout_Duration:Bout_Duration*2],bouts_array_flat[i,Bout_Duration*2:] = sub_x, sub_y, sub_body_angle

                    i = i+1
                    onset.append(id_st)
                    offset.append(id_ed)
                    
        bouts_array = bouts_array[:i,:,:]
        bouts_array_flat = bouts_array_flat[:i,:]

        return onset,offset,bouts_array,bouts_array_flat

    return segment_from_mobility


def create_segmentation_from_code(Min_Code_Ampl=1,SpikeDist=120,Bout_Duration=140):

    def segment_from_code(z,tail_angle):

        # FINDING PEAKS IN SPARSE CODE:
        z_max = np.max(np.abs(z),axis=1)
        peaks, _ = find_peaks(z_max, height=Min_Code_Ampl,distance=SpikeDist)
        peaks_bin = np.zeros(tail_angle.shape[0])
        peaks_bin[peaks]=1

        '''kernel = np.ones(SpikeDist)
        filtered_forward = np.convolve(kernel,peaks_bin, mode='full')[:peaks_bin.shape[0]]
        is_tail_active = 1.0*(filtered_forward>0)'''

        # EXTRACT BOUTS:
        bouts_array = np.zeros((len(peaks),Bout_Duration,7))
        bouts_hat_array = np.zeros((len(peaks),Bout_Duration,7))

        onset = []
        offset = []
        aligned_peaks = []

        i = 0
        Margin_before_peak = 0

        for iter_,peak in enumerate(peaks):
                if ((peak>Margin_before_peak)&(peak+140<tail_angle.shape[0]-Margin_before_peak)):
                    id_st = peak - Margin_before_peak
                    id_ed = id_st +140
                    tmp = tail_angle[id_st:id_ed,7]

                    try:
                        peak_location = align_bout_peaks(tmp,quantile_threshold = 0.25 , minimum_peak_size = 0.25, minimum_peak_to_peak_amplitude = 4,debug_plot_axes=None)
                    except:
                        peak_location = np.nan
                    if np.isnan(peak_location):
                        peak_location = peak
                    else:
                        aligned_peaks.append(id_st+peak_location)
                        id_st = id_st+peak_location - Margin_before_peak -20
                        id_ed = id_st +140
                    bouts_array[i,:,:] = tail_angle[id_st:id_ed,:7]
                    i = i+1
                    onset.append(id_st)
                    offset.append(id_ed)
        bouts_array = bouts_array[:i,:,:]

        return onset,offset,bouts_array

    return segment_from_code







