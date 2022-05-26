
import numpy as np
from segmentation.threshold import estimate_threshold_using_GMM
import scipy.signal as signal
import pandas as pd
from utils.utils import find_onset_offset_numpy

def remove_slow_trend(tail_angle,ref_segment=7):

    tail_angle_detrend = np.zeros_like(tail_angle) 
    x = np.copy(tail_angle[:,ref_segment])

    # Compute tail active
    trend = signal.savgol_filter(x, window_length=31, polyorder=1, deriv=0, delta=1.0, axis=- 1, mode='interp', cval=0.0)
    detrend = np.abs(x-trend)
    x = pd.DataFrame({'x':detrend})
    win = 15
    x = x.rolling(win).mean().values[:,0]
    BT,ax = estimate_threshold_using_GMM(x,margin_std=2.5,axis=None)

    baseline = np.zeros_like(x)
    tail_active = (x>BT)
    kernel = np.ones(40)
    filtered_timeforward = np.convolve(kernel,tail_active, mode='full')[:tail_active.shape[0]] # Trick to make convolution causal
    tail_active = 1.0*((filtered_timeforward)>0)#*filtered_timebackward)>0)

    # Remove too short activation:
    onset,offset,duration = find_onset_offset_numpy(tail_active)
    tail_active = 0*tail_active
    for on_,off_,dur_ in zip(onset,offset,duration):
        if dur_>60:
            tail_active[on_:off_]=1

    tail_angle_detrend = np.zeros_like(tail_angle)
    for s in range(tail_angle.shape[1]):
        baseline = np.zeros_like(tail_active)
        baseline[tail_active==0]=tail_angle[tail_active==0,s]
        onset,offset,duration = find_onset_offset_numpy(tail_active)
        for on_,off_,dur_ in zip(onset,offset,duration):
            tmp = tail_angle[on_:off_,s]
            if dur_>1:
                baseline[on_:off_] = tmp[0]#np.linspace(tmp[0],tmp[-1],dur_)
            else:
                baseline[on_] = tail_angle[on_,s]
        tail_angle_detrend[:,s] = tail_angle[:,s]-baseline

    return tail_angle_detrend