from scipy.special import binom
import numpy as np
import scipy.signal as signal
import pandas as pd

import os
import json

# Data Wrangling
import h5py
import numpy as np
import pandas as pd


from scipy.ndimage.filters import maximum_filter1d 
import scipy.signal as signal
from scipy.special import binom
from scipy.signal import savgol_filter

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt




def max_filter1d_valid(a, W):
    hW = (W-1)//2 # Half window size
    return maximum_filter1d(a,size=W,mode='reflect')#[hW:-hW]

# Fast Derivative Computation:
from scipy.special import binom
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

from sklearn.decomposition import PCA


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

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

def batch_tail_angle(tail_angle,batch_duration=700*30):
    N = int(np.ceil(tail_angle.shape[0]/(batch_duration)))
    Nseg = tail_angle.shape[1]
    tail_angle_ = np.zeros((N*batch_duration,Nseg))
    tail_angle_[:tail_angle.shape[0],:] = tail_angle
    tail_angle_batch =  tail_angle_.reshape(N,batch_duration,Nseg)
    tail_angle_batch = np.swapaxes(tail_angle_batch,0,1)
    tail_angle_batch = np.swapaxes(tail_angle_batch,1,2)
    return tail_angle_batch




def estimate_threshold_using_GMM(x,margin_std,axis=None):
    
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
##########################################   TAIL CLASS    ###########################################
######################################################################################################

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    #y = signal.filtfilt(b, a, data)
    y = signal.lfilter(b,a,data)
    return y

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
    
    def remove_baseline(self,N = 70):
        ### Substract Baseline from tail angle:
        tail_angle_no_baseline = np.zeros_like(self.angle_smooth)
        win = np.ones(N)/N
        for i in range(self.angle_smooth.shape[-1]):
            x = self.angle_smooth[:,i]
            tail_angle_no_baseline[:,i] = x[:] - np.convolve(x,win, mode='full')[:x.shape[0]]
        return tail_angle_no_baseline

    '''def remove_baseline(self,cutoff_freq=2,order=5):
        ### Substract Baseline from tail angle:
        tail_angle_no_baseline = np.zeros_like(self.angle_smooth)
        for i in range(self.angle_smooth.shape[-1]):
            x = self.angle_smooth[:,i]
            #tail_angle_no_baseline[:,i] = x[:] - np.convolve(x,win, mode='full')[:x.shape[0]]
            tail_angle_no_baseline[:,i] = butter_highpass_filter(x,cutoff_freq,self.fps,order)

        return tail_angle_no_baseline'''






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



def display_trajectory(df,index,past_memory=3*700):
    #set up the figure
    fig = plt.figure(figsize=(5,5))
    canvas_width, canvas_height = fig.canvas.get_width_height()
    ax = fig.add_subplot()
    circle = plt.Circle((0, 0),25, ec='r', fill=False)
    ax.add_artist(circle)

    df_past = df.iloc[max(0,index-past_memory):index+1]
    ax.scatter(df_past['x'], df_past['y'], c = 'r', s =1,alpha=0.1)
    for i in range(0,df_past.shape[0],100):
        c = 1*np.cos(df_past.iloc[i]['angle'])
        s = 1*np.sin(df_past.iloc[i]['angle'])
        x_end = df_past.iloc[i]['x']
        y_end = df_past.iloc[i]['y']
        ax.arrow(x_end,y_end,c,s, head_width=1, head_length=1, fc='k', ec='k')

    c = 3*np.cos(df.iloc[index]['angle'])
    s = 3*np.sin(df.iloc[index]['angle'])
    x_end = df.iloc[index]['x']
    y_end = df.iloc[index]['y']
    ax.arrow(x_end,y_end,c,s, head_width=1, head_length=1, fc='b', ec='b')
    ax.set_xlim(-25,25)
    ax.set_ylim(-25,25)
    return fig

