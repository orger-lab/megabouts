import numpy as np
from scipy.signal import find_peaks
from segmentation.align import align_bout_peaks

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