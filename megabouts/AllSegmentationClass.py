import os
import json

# Data Wrangling
import h5py
import numpy as np
import pandas as pd


from scipy.special import binom
import scipy.signal as signal
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

######################################################################################################
########################################## USEFUL FUNCTION ###########################################
######################################################################################################

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


def clean_using_pca(X,num_pcs=4):
    
    # X Should be NumBouts,T,NumSegments
    T=X.shape[0]
    NumSeg=X.shape[1]

    pca = PCA(n_components=num_pcs)
    pca.fit(X)
    low_D = pca.transform(X)

    X_clean = pca.inverse_transform(low_D)
    
    return X_clean


def compute_smooth_tail_angle(tail_angle,thresh_error):

    # Define Cumul Sum and Find Tracking Nan:
    tail_angle[tail_angle<thresh_error]=0
    cumul_tail_angle=np.cumsum(tail_angle,1)
    notrack=np.where(np.sum(cumul_tail_angle,1)==0)[0]

    smooth_cumul_tail_angle=np.copy(cumul_tail_angle)
    for n in range(1,cumul_tail_angle.shape[1]-1):
        smooth_cumul_tail_angle[:,n]=np.mean(cumul_tail_angle[:,n-1:n+2],1)

    for n in range(cumul_tail_angle.shape[1]):
        tmp=cumul_tail_angle[:,n]
        tmp=signal.savgol_filter(tmp, 11, 2, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        smooth_cumul_tail_angle[:,n]=tmp
        
    return cumul_tail_angle,smooth_cumul_tail_angle,notrack


######################################################################################################
##########################################   TAIL CLASS    ###########################################
######################################################################################################

class Tail(object):
    
    def __init__(self,
                 relative_angle, # Input shouldn't be cumulated
                 fps = 700,
                 Reference_tail_segment_start = 2,
                 Reference_tail_segment_end = 7):
        

        self.fps = fps
        self.Reference_tail_segment_start = Reference_tail_segment_start
        self.Reference_tail_segment_end = Reference_tail_segment_end
        self.relative_angle = relative_angle
        self.T = self.relative_angle.shape[0]
        
        # Make sure the input uses -99 as flag for nan:
        id = np.where(np.isnan(np.sum(self.relative_angle,axis=1)))[0]
        self.relative_angle[id,:]=-99
        

        self.angle = None
        self.angle_smooth = None
        self.angle_speed = None
        self.notrack_mask = np.zeros(self.T)
        self.notrack_id = None

    def tail_angle_preprocessing(self,NumPCs=4,N_Filt_diff=71,thresh_error=-50):

        # Smooth Tail angle and compute measure of intensity
        cumul_tail_angle,smooth_cumul_tail_angle,notrack = compute_smooth_tail_angle(self.relative_angle,thresh_error=thresh_error)
        self.notrack_id = notrack
        self.notrack_mask[self.notrack_id] = 1 

        self.angle = cumul_tail_angle
        self.angle_smooth = smooth_cumul_tail_angle

        # Clean using PCA:   
        if NumPCs>0:
            X = self.angle_smooth
            X[self.notrack_mask==0] = clean_using_pca(self.angle_smooth[self.notrack_mask==0])
            self.angle_smooth = X
        
        tail_angle_speed = np.zeros_like(smooth_cumul_tail_angle)
        for s in range(tail_angle_speed.shape[1]):
            tail_angle_speed[:,s] = diff_but_better(smooth_cumul_tail_angle[:,s],dt=1/self.fps, filter_length=N_Filt_diff)
        
        self.angle_speed = tail_angle_speed

######################################################################################################
##########################################   TAIL ACTIVE   ###########################################
######################################################################################################


from scipy.ndimage.filters import maximum_filter1d 

def max_filter1d_valid(a, W):
    hW = (W-1)//2 # Half window size
    return maximum_filter1d(a,size=W,mode='reflect')#[hW:-hW]

def medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)


def mexican_hat_filter(x,MinFiltSize,MaxFiltSize):
    # max filter flattens out beat variations (timescale needs to be adjusted to be larger than interbeat and smaller than the bout length)
    max_filt=max_filter1d_valid(x,MaxFiltSize)
    # min filter removes baseline fluctuations (needs to be well above the bout length)
    min_filt=-max_filter1d_valid(-x,MinFiltSize)
    y=max_filt-min_filt
    return y,max_filt,min_filt


from sklearn.mixture import GaussianMixture
from scipy import stats

class Binarization(object):

    def __init__(self, tail):

        self.tail = tail
        self.low_pass_speed = None
        self.smooth_residual = None

    # Compute low pass tail speed:
    def compute_low_pass_speed(self,win_BC = 10,MinFiltSize=400,MaxFiltSize=20):
        
        # Compute difference in segment angles, because we want to detect tail movement
        tmp=np.diff(self.tail.angle_smooth,axis=0)
        z=np.zeros((1,tmp.shape[1]))
        speed_tail_angle=np.vstack((z,tmp))
        speed_tail_angle[self.tail.notrack_id,:]=0
        
        # Interpolate on NoTrack
        if len(self.tail.notrack_id)>0:
            if self.tail.notrack_id[-1]==self.tail.T-1:
                speed_tail_angle[self.tail.notrack_id[:-1]+1,:]=0
            else:
                speed_tail_angle[self.tail.notrack_id+1,:]=0
        speed_tail_angle = speed_tail_angle[:,0:self.tail.Reference_tail_segment_end]
        
        # Smooth the tail movement
        filtered_speed_tail_angle=np.zeros_like(speed_tail_angle)
        for i in range(speed_tail_angle.shape[1]):
            filtered_speed_tail_angle[:,i]= signal.convolve(
                speed_tail_angle[:,i],
                1/win_BC*signal.boxcar(M=win_BC,sym=False),
                mode='same')

        # Sum the angle differences down the length of the tail to give prominence to regions of continuous curvature in one direction
        cumul_filtered_speed=np.cumsum(filtered_speed_tail_angle,1)

        # Sum the absolute value of this accumulated curvature so bends in both directions are considered
        super_cumul = np.cumsum(np.abs(cumul_filtered_speed),1)
        smooth_tail_speed= signal.convolve(
                super_cumul[:,-1],
                1/win_BC*signal.boxcar(M=win_BC,sym=False),
                mode='same')

        low_pass_tail_speed,max_filt,min_filt = mexican_hat_filter(smooth_tail_speed,MinFiltSize,MaxFiltSize)

        self.low_pass_speed = low_pass_tail_speed
    
    def compute_noiselevel(self,win_zscore = 30,win_smoothing = 15, win_med=51):

        # Compute rolling z-score:
        z_score_rolling = np.zeros(self.tail.T)
        for i in range(self.tail.Reference_tail_segment_end):
            ts = pd.Series(self.tail.angle_smooth[:,i])
            tmp = (ts-ts.rolling(window=win_zscore).mean())/ts.rolling(window=win_zscore).std()
            z_score_rolling = z_score_rolling + tmp.values
        
        # Compute residual with smooth version as measure of noise:
        x_smooth = savgol_filter(z_score_rolling,win_smoothing, 2)        
        residual = np.power(z_score_rolling - x_smooth,2)
        self.smooth_residual = medfilt(residual,win_med)


    def estimate_threshold_using_GMM(self,x,margin_std,axis=None):
        
        log_x = np.log(x[x>0])

        X = log_x[:,np.newaxis]
        gm = GaussianMixture(n_components=2, random_state=0).fit(X)

        weights = gm.weights_
        means = gm.means_
        covars = gm.covariances_

        id = np.argmin(means)
        sigma  = np.sqrt(covars[id]) # Standard Deviation
        BoutThresh = np.exp(means[id] + margin_std*sigma)[0]
        f_axis = log_x.copy().ravel()
        f_axis.sort()
        if axis is not None:
            axis.hist(log_x, bins=1000, histtype='bar', density=True, ec='red', alpha=0.1)
            #axis.plot(bins,count)
            axis.plot(f_axis,weights[0]*stats.norm.pdf(f_axis,means[0],np.sqrt(covars[0])).ravel(), c='blue')
            axis.plot(f_axis,weights[1]*stats.norm.pdf(f_axis,means[1],np.sqrt(covars[1])).ravel(), c='blue')
            axis.plot(f_axis,weights[0]*stats.norm.pdf(f_axis,means[0],np.sqrt(covars[0])).ravel()+weights[1]*stats.norm.pdf(f_axis,means[1],np.sqrt(covars[1])).ravel(),
            c='green')
            axis.scatter(means[id] + margin_std*sigma,0,s=100,c='r',marker=">")

        return BoutThresh[0],axis

######################################################################################################
##########################################    HALF BEAT    ###########################################
######################################################################################################

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
    

def refine_segmentation_bouts(onset,offset,Min_Duration, Min_IBI):
    
    duration = offset - onset
    onset = onset[duration>Min_Duration]
    offset = offset[duration>Min_Duration]
    duration = duration[duration>Min_Duration]
    inter_bouts = [on_-off_ for (on_,off_) in zip(onset[1:],offset[:-1])]
    # Merge Bouts with too small inter_bouts_interval:
    id = np.where(np.array(inter_bouts)<Min_IBI)[0]
    # remove offset of id and onset of id+1
    offset = np.delete(offset,id)
    onset = np.delete(onset,id+1)
    duration = offset-onset

    inter_bouts = [on_-off_ for (on_,off_) in zip(onset[1:],offset[:-1])]
    duration = offset-onset
    return onset,offset,duration,inter_bouts


import cv2
from skimage import measure
from skimage import filters
from skimage.measure import label, regionprops,regionprops_table

def find_half_beat(bout_slice,Half_BCFilt = 150, stdThres = 5,MinSizeBlob = 500):

    BCFilt = Half_BCFilt*2+1
    X = bout_slice
    oversample_slice = cv2.resize(X, dsize=(X.shape[1],X.shape[0]*10), interpolation=cv2.INTER_CUBIC)
        
    filtered_=np.zeros_like(oversample_slice)
    binary_thresh_up=np.zeros_like(oversample_slice)
    binary_thresh_down=np.zeros_like(oversample_slice)
    for i in range(oversample_slice.shape[1]):
            filtered_[:,i]= signal.convolve(oversample_slice[:,i],1/BCFilt*signal.boxcar(M=BCFilt,sym=True),mode='same')
            sigma = np.std(filtered_[:,i])
            binary_thresh_up[:,i] = filtered_[:,i]+sigma/stdThres
            binary_thresh_down[:,i] = filtered_[:,i]-sigma/stdThres

    binary_image = np.zeros_like(oversample_slice)
    for i in range(oversample_slice.shape[1]):
        binary_image[:,i] = (oversample_slice[:,i]>binary_thresh_up[:,i]) + -1*(oversample_slice[:,i]<binary_thresh_down[:,i])

    all_labels_pos = measure.label((binary_image)==1)
    if len(np.where(all_labels_pos)[0])>0:
        props_pos = regionprops_table(all_labels_pos, properties=('area','centroid'))#,
        half_beat_pos = []
        for i,lab_ in enumerate(np.unique(all_labels_pos)[1:]):
            if props_pos['area'][i]> MinSizeBlob:
                id = np.where(all_labels_pos[:,-1]==lab_)[0]
                if len(id)>0:
                    half_beat_pos.append(id[np.argmax(oversample_slice[id,-1])])
    else:
        half_beat_pos = []

    all_labels_neg = measure.label((binary_image)==-1)
    if len(np.where(all_labels_neg)[0])>0:
        props_neg = regionprops_table(all_labels_neg, properties=('area','centroid'))
        half_beat_neg = []
        for i,lab_ in enumerate(np.unique(all_labels_neg)[1:]):
            if props_neg['area'][i]> MinSizeBlob:
                id = np.where(all_labels_neg[:,-1]==lab_)[0]
                if len(id)>0:
                    half_beat_neg.append(id[np.argmin(oversample_slice[id,-1])])
    else:
        half_beat_neg = []

    half_beat_pos = np.floor(np.array(half_beat_pos)/10).astype('int')
    half_beat_neg = np.floor(np.array(half_beat_neg)/10).astype('int')

    return half_beat_pos,half_beat_neg,binary_image


class HalfBeat(object):
    
    def __init__(self,
                 tail_active, # Input shouldn't be cumulated
                 tail):
        
        self.tail = tail
        self.tail_active_in = tail_active
        self.onset = None
        self.offset = None
        self.tail_active_out = None
        self.half_beat_pos = None
        self.half_beat_neg = None

    def find_slice(self,Min_Duration = 80,Min_IBI = 10):

        # Compute Running Average to compute bouts amplitude:
        onset,offset,duration = find_onset_offset_numpy(self.tail_active_in==1)
        # Refine Segmentation:
        onset, offset, duration, inter_bouts = refine_segmentation_bouts(onset, offset, Min_Duration, Min_IBI)
        onset, offset = onset.tolist(),offset.tolist()
        self.onset = onset
        self.offset = offset
        self.tail_active_out = np.zeros_like(self.tail_active_in)
        for on_,off_ in zip(onset,offset):
            self.tail_active_out[on_:off_]=1

    
    def extract_half_beat(self,Half_BCFilt = 150, stdThres = 5,MinSizeBlob=500,Margin=10):

        peaks_pos = []
        peaks_neg = []

        for i,(on_,off_) in enumerate(zip(self.onset,self.offset)):
            on_ = max(0,on_ - Margin)
            off_ = min(off_ + Margin,self.tail.T)
            bout_slice = self.tail.angle_smooth[on_:off_,self.tail.Reference_tail_segment_start:self.tail.Reference_tail_segment_end+1]
            half_beat_pos,half_beat_neg,binary_image = find_half_beat(bout_slice,Half_BCFilt = Half_BCFilt, stdThres = stdThres,MinSizeBlob=MinSizeBlob)
            
            peaks_pos = peaks_pos + (half_beat_pos+on_).tolist()
            peaks_neg = peaks_neg + (half_beat_neg+on_).tolist()

        peaks_pos = np.array(peaks_pos)
        peaks_neg = np.array(peaks_neg)

        self.half_beat_pos = peaks_pos
        self.half_beat_neg = peaks_neg

        

######################################################################################################
##########################################   BREAKPOINT    ###########################################
######################################################################################################



