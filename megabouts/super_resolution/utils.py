from scipy.special import binom
import numpy as np
import scipy.signal as signal
import pandas as pd

#from numba import njit, prange
'''
import ray
ray.init(num_cpus = 60)


@ray.remote
def work_find_nearest_bouts_parallel(sub_x,sub_y,sub_body_angle,ref_bouts_flat,ref_labels):
    #sub_x,sub_y,sub_body_angle = x[on_-20:on_+60],y[on_-20:on_+60],body_angle[on_-20:on_+60]
    Pos = np.zeros((2,80))
    Pos[0,:] = sub_x-sub_x[0]
    Pos[1,:] = sub_y-sub_y[0]
    theta=-sub_body_angle[0]
    body_angle_rotated=sub_body_angle-sub_body_angle[0]
    RotMat=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    PosRot=np.dot(RotMat,Pos)
    sub_x,sub_y,sub_body_angle = PosRot[0,:],PosRot[1,:],body_angle_rotated
    traj = np.hstack((sub_x,sub_y,sub_body_angle))
    diff = ref_bouts_flat - traj
    error = np.sqrt(np.sum(np.power(diff,2),axis=1))
    i_m = np.argsort(error)[0]
    bout_cat  = ref_labels[i_m]

    return bout_cat

def find_nearest_bouts_parallel(onset,x,y,body_angle,ref_bouts_flat,ref_labels):
    
    ref_bouts_flat_id = ray.put(ref_bouts_flat)
    ref_labels_id = ray.put(ref_labels)
    #x_id =  ray.put(x)
    #y_id =  ray.put(y)
    #body_angle_id =  ray.put(body_angle)
    #result_ids = [work_find_nearest_bouts_parallel.remote(on_,x_id,y_id,body_angle_id,ref_bouts_flat_id,ref_labels_id) for on_ in onset]
    result_ids = [work_find_nearest_bouts_parallel.remote(x[on_-20:on_+60],y[on_-20:on_+60],body_angle[on_-20:on_+60],ref_bouts_flat_id,ref_labels_id) for on_ in onset]
    #result_ids = [work_find_nearest_bouts_parallel.remote(x[on_-20:on_+60],y[on_-20:on_+60],body_angle[on_-20:on_+60],ref_bouts_flat,ref_labels) for on_ in onset]
    results = ray.get(result_ids)

    return results
'''



from scipy.ndimage.filters import maximum_filter1d 
import scipy.signal as signal

def max_filter1d_valid(a, W):
    hW = (W-1)//2 # Half window size
    return maximum_filter1d(a,size=W,mode='reflect')#[hW:-hW]


def mexican_hat_tail_speed(smooth_tail_speed,MinFiltSize,MaxFiltSize):
    # max filter flattens out beat variations (timescale needs to be adjusted to be larger than interbeat and smaller than the bout length)
    max_filt=max_filter1d_valid(smooth_tail_speed,MaxFiltSize)
    # min filter removes baseline fluctuations (needs to be well above the bout length)
    min_filt=-max_filter1d_valid(-smooth_tail_speed,MinFiltSize)
    
    low_pass_tail_speed=max_filt-min_filt

    return low_pass_tail_speed,max_filt,min_filt

def estimate_speed_threshold(speed,margin_std, bin_log_min = -5, bin_log_max=5 ):
    
    log_speed = np.log(speed)
    count,edge = np.histogram(log_speed,np.arange(bin_log_min,bin_log_max,0.1))
    bins = (edge[:-1]+edge[1:])/2
    # Append Far Value at begining because find_peak does not detect first peak
    count = np.concatenate((np.array([count[-1]]),count))
    peaks = signal.find_peaks(count, distance=5)[0] # distance correspond to 5 bin in log space
    count = count[1:]
    peaks = peaks-1
    # full width at half maximum of the noise peak
    noise_peak_loc = peaks[0]
    # Check if first value above or beyon half max:
    if count[0]>count[noise_peak_loc]/2:
        # Estimate Half Width at half maximum:
        w = np.where(count<count[noise_peak_loc]/2)[0][0]
        fwhm = 2*np.exp(bins[int(np.round(w))])
    else:
        w = np.where(count[noise_peak_loc:]<count[noise_peak_loc]/2)[0][0]
        w = int(w)
        fwhm = 2*np.exp(bins[w])

    sigma = fwhm/2.355 # According to the relation between fwhm and std for gaussian
    BoutThresh = np.exp(bins[noise_peak_loc]) + margin_std*sigma

    return BoutThresh
'''
@njit(parallel=True)
def find_nearest_bouts(onset,x,y,body_angle,ref_bouts_flat,ref_labels):
    bout_cat = -np.ones(len(onset))

    for i in prange(len(onset)):

        on_ = onset[i]
        sub_x,sub_y,sub_body_angle = x[on_-20:on_+60],y[on_-20:on_+60],body_angle[on_-20:on_+60]
        Pos = np.zeros((2,80))
        Pos[0,:] = sub_x-sub_x[0]
        Pos[1,:] = sub_y-sub_y[0]
        theta=-sub_body_angle[0]
        body_angle_rotated=sub_body_angle-sub_body_angle[0]
        RotMat=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        PosRot=np.dot(RotMat,Pos)
        sub_x,sub_y,sub_body_angle = PosRot[0,:],PosRot[1,:],body_angle_rotated

        traj = np.hstack((sub_x,sub_y,sub_body_angle))
        diff = ref_bouts_flat - traj
        error = np.sqrt(np.sum(np.power(diff,2),axis=1))
        i_m = np.argsort(error)[0]
        bout_cat[i]=ref_labels[i_m]

    return bout_cat
'''

import matplotlib.pyplot as plt
import h5py
from scipy.signal import find_peaks

DEBUG_TEXT = False
DEBUG_PLOT = False
def debug(name,value=''):
    """prints variables and their contents when debuging"""
    if(DEBUG_TEXT):
        print(f'{name} : {value}')

def debug_plot( ax, 
                bout,bout_cumsum, 
                max_location_bout, min_location_bout,
                max_location_cumsum, min_location_cumsum,
                peak_search_start, peak_search_end, peak_location):
    """plots all relevat debug information"""
    if(ax != None):
        ax.plot(bout)
        ax.plot(bout_cumsum)
        # bout peaks
        ax.plot(max_location_bout , bout[max_location_bout],'+b')
        ax.plot(min_location_bout , bout[min_location_bout],'+r')
        # cumsum peaks
        ax.plot(max_location_cumsum , bout_cumsum[max_location_cumsum],'xb')
        ax.plot(min_location_cumsum , bout_cumsum[min_location_cumsum],'xr')
        # search range
        ax.plot([peak_search_start,peak_search_start] ,[-2,2],color=[1,0,1],linewidth=2)
        ax.plot([peak_search_end,peak_search_end] ,[-2,2],color=[0,1,1],linewidth=2)
        # selected peak
        if peak_location is np.NaN:
            ax.text(0,1,'ERROR',color=[1,1,1],backgroundcolor=[1,0,0],horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
        else:
            ax.plot(peak_location ,bout[peak_location],'og',ms = 10,mfc='none')

def align_bout_peaks(bout_data,quantile_threshold = 0.25 , minimum_peak_size = 0.25, minimum_peak_to_peak_amplitude = 4,debug_plot_axes=None):

    # only use the first 150 segments to calculate the alignment.
    bout = bout_data[:150]   
    # 1. Find peaks in data and remove peaks with small amplitude
    
    max_location_bout, _ = find_peaks(bout , height = minimum_peak_size)
    min_location_bout, _ = find_peaks(bout*-1, height = minimum_peak_size)
    peaks_bout = np.sort(np.concatenate((max_location_bout,min_location_bout)))
    debug('peak locations',peaks_bout)
    # 2. Smooth trace by zeroing small traces and calculate cumsum of the smoothed data
    tmp = bout
    threshold = np.quantile(abs(tmp), quantile_threshold)
    tmp[abs(tmp) < threshold] = 0
    bout_cumsum = np.cumsum(tmp)
    # 3. Find peaks in cumsum data
    max_location_cumsum, _ = find_peaks(bout_cumsum)
    min_location_cumsum, _ = find_peaks(bout_cumsum*-1)
    peaks_cumsum = np.sort(np.concatenate((max_location_cumsum,min_location_cumsum)))
    debug('peak locations cumsum',peaks_cumsum)
    peak_search_start = np.NaN
    peak_search_end = np.NaN
    # If there are peaks in the cumsum data, find the optimal place to search for the peak in the bout data
    if any(peaks_cumsum):
        # calculate the distances between consecutive peaks of the cumsum trace
        distances = np.abs(np.diff(np.pad(bout_cumsum[peaks_cumsum],(1,0))))
        # find first peak whose distance between peaks is bigger than the threshold
        idx = np.where(distances > minimum_peak_to_peak_amplitude)[0]
        
        debug('peak to peak amplitudes',distances)
        debug('first big peak index',idx)

        # skip the algorithm and throw error if there are no peaks in the cumsum data
        if idx.size != 0:
            idx = idx[0]
            # if there the first peak of the cumsum data is selected, check if there is any peak in the bout data before that peak
            if idx == 0:
                # if there is any peak in the bout data before the peak in the cumsum data,
                # set the search boundaries to be between the beginning of the data and the location of the peak in the cumsum data
                # if there aren't any peaks,
                # set the search boundaries to be between the location of the first and second peaks in the cumsum data
                debug('points before peak',np.where(peaks_bout < peaks_cumsum[0])[0])
                if len(np.where(peaks_bout < peaks_cumsum[0])[0]) > 0:
                    peak_search_start = 0
                    peak_search_end   = peaks_cumsum[0]
                else :
                    peak_search_start = peaks_cumsum[0]
                    peak_search_end   = peaks_cumsum[1]
            else:
                # if any other peak is selected, search between the selected point and the one before
                peak_search_start = peaks_cumsum[idx-1]
                peak_search_end   = peaks_cumsum[idx]
            debug('search start', peak_search_start)
            debug('search end',peak_search_end)
            # 4. Delete peaks outside the search range
            peaks_bout = peaks_bout[(peaks_bout > peak_search_start) & (peaks_bout < peak_search_end)]
        else:
            peaks_bout = []

    if any(peaks_bout):
        peak_location=peaks_bout[0]
        debug('selected peak',peak_location)
    else:
        peak_location = np.NaN
        print('Failed to find a peak within these constraints')

    if(DEBUG_PLOT):
        debug_plot(debug_plot_axes,
                    bout,bout_cumsum, 
                    max_location_bout, min_location_bout,
                    max_location_cumsum, min_location_cumsum,
                    peak_search_start, peak_search_end, peak_location)
    return peak_location




def diff_but_better(x,dt=1/700, filter_length=71):
    if not filter_length % 2 == 1:
        raise ValueError('Filter length must be odd.')
    M = int((filter_length - 1) / 2)
    m = int((filter_length - 3) / 2)
    coefs = [(1 / 2**(2 * m + 1)) * (binom(2 * m, m - k + 1) - binom(2 * m, m - k - 1))
        for k in range(1, M + 1)]
    coefs = np.array(coefs)
    kernel = np.concatenate((coefs[::-1],[0],-coefs))
    filtered = np.convolve(kernel,x, mode='valid')
    filtered = (1 / dt) * filtered
    filtered = np.concatenate((np.nan*np.ones(M),filtered,np.nan*np.ones(M)))
    return filtered

    
def find_onset_offset_numpy(binary_serie):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(binary_serie,1).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    onset = ranges[:,0]
    offset = ranges[:,1]
    duration = offset-onset
    return onset,offset,duration

# Cleaning trajectory using Alexandre Algorithm
def one_euro_filter(x,fc_min,beta,rate):
    dx = 0
    x_smooth = np.zeros_like(x)
    x_smooth[0] = x[0]

    fc = fc_min
    tau = 1/(2*np.pi*fc)
    te=1/rate
    alpha = 1/(1+tau/te)
    
    for i in range(1,len(x)):
                
        x_smooth[i] = alpha * x[i] + (1-alpha) * x_smooth[i-1]
        
        x_dot = (x_smooth[i]-x_smooth[i-1])*rate
        fc = fc_min + beta*np.abs(x_dot)
        tau = 1/(2*np.pi*fc)
        te=1/rate
        alpha = 1/(1+tau/te)

    return x_smooth

