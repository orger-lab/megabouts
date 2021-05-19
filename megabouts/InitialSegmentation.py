import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import stats
import cv2
import scipy.signal as signal

from skimage import measure
from skimage import filters
from skimage.measure import label, regionprops,regionprops_table


from utils.utils_bouts import diff_but_better,compute_smooth_tail_angle,compute_tail_speed,mexican_hat_tail_speed,estimate_speed_threshold,find_onset_offset_numpy
from utils.utils_beat import refine_segmentation_bouts,find_zeros_crossing,is_there_oscillation,clean_peak_forcing_distance,clean_peak_forcing_alternation,interleave,is_break_in_amplitude,is_break_in_interbeatinterval,is_there_signal


def tail_angle_preprocessing(tail_angle,NumSegments=7,BCFilt=10,MinFiltSize=400,MaxFiltSize=20):

    # Smooth Tail angle and compute measure of intensity
    cumul_tail_angle,smooth_cumul_tail_angle,notrack = compute_smooth_tail_angle(tail_angle)
    smooth_tail_speed,speed_tail_angle,super_cumul = compute_tail_speed(smooth_cumul_tail_angle,notrack,NumSegments,BCFilt)
    
    # MinFilt remove unstable baseline & MaxFilt merge bouts
    low_pass_tail_speed,max_filt,min_filt = mexican_hat_tail_speed(smooth_tail_speed,MinFiltSize,MaxFiltSize)
    
    # Compute Speed:
    tail_angle_speed = np.zeros_like(smooth_cumul_tail_angle)

    for s in range(tail_angle_speed.shape[1]):
        tail_angle_speed[:,s] = diff_but_better(smooth_cumul_tail_angle[:,s],dt=1/700, filter_length=71)
    
    return smooth_cumul_tail_angle,low_pass_tail_speed,tail_angle_speed,notrack

def estimate_speed_threshold_using_GMM(speed,margin_std,axis=None):
    
    log_speed = np.log(speed[speed>0])
    #bin_log_min = -5
    #bin_log_max = 5
    #count,edge = np.histogram(log_speed,np.arange(bin_log_min,bin_log_max,0.1))
    #bins = (edge[:-1]+edge[1:])/2

    X = log_speed[:,np.newaxis]
    gm = GaussianMixture(n_components=2, random_state=0).fit(X)

    weights = gm.weights_
    means = gm.means_
    covars = gm.covariances_

    id = np.argmin(means)
    sigma  = np.sqrt(covars[id]) # Standard Deviation
    BoutThresh = np.exp(means[id] + margin_std*sigma)[0]
    f_axis = log_speed.copy().ravel()
    f_axis.sort()
    if axis is not None:
        axis.hist(log_speed, bins=1000, histtype='bar', density=True, ec='red', alpha=0.1)
        #axis.plot(bins,count)
        axis.plot(f_axis,weights[0]*stats.norm.pdf(f_axis,means[0],np.sqrt(covars[0])).ravel(), c='blue')
        axis.plot(f_axis,weights[1]*stats.norm.pdf(f_axis,means[1],np.sqrt(covars[1])).ravel(), c='blue')
        axis.plot(f_axis,weights[0]*stats.norm.pdf(f_axis,means[0],np.sqrt(covars[0])).ravel()+weights[1]*stats.norm.pdf(f_axis,means[1],np.sqrt(covars[1])).ravel(),
        c='green')

    return BoutThresh[0],axis



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
                #np.argmax(res[id,8])
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
                #np.argmax(res[id,8])
                if len(id)>0:
                    half_beat_neg.append(id[np.argmin(oversample_slice[id,-1])])
    else:
        half_beat_neg = []

    half_beat_pos = np.floor(np.array(half_beat_pos)/10).astype('int')
    half_beat_neg = np.floor(np.array(half_beat_neg)/10).astype('int')

    return half_beat_pos,half_beat_neg,binary_image


class InitialSegmentation:
    
    def __init__(self,
                 tail_angle, # Input shouldn't be cumulated
                 fps = 700,
                 Reference_tail_segment = 6,
                 BCFilt = 10,
                 MinFiltSize = 400,
                 MaxFiltSize = 20,
                 Margin_std_noise = 2.2
                 ):
        
        self.fps = fps
        self.Reference_tail_segment = Reference_tail_segment
        self.Margin_std_noise = Margin_std_noise

        self.tail_angle = tail_angle

        smooth_cumul_tail_angle,low_pass_tail_speed,tail_angle_speed,notrack = tail_angle_preprocessing(tail_angle,NumSegments=Reference_tail_segment,BCFilt=BCFilt,MinFiltSize=MinFiltSize,MaxFiltSize=MaxFiltSize)

        self.tail_angle_smooth = smooth_cumul_tail_angle
        self.tail_angle_speed_low_pass = low_pass_tail_speed

        self.BoutThresh = None
        self.initial_onset = None
        self.initial_offset = None
        self.initial_tail_active = None
        
        self.half_beat_pos = None
        self.half_beat_neg = None

    def find_activity_threshold(self,axis):
        # Compute threshold based on FWHM of speed distribution
        BoutThresh,axis = estimate_speed_threshold_using_GMM(self.tail_angle_speed_low_pass,self.Margin_std_noise,axis=axis)
        self.BoutThresh = BoutThresh
        return BoutThresh,axis
    
    def initial_segmentation(self,Min_Duration = 80,Min_IBI = 10):
        if self.BoutThresh==None:
            raise Exception('You should first compute the activity threshold using self.find_activity_threshold')
        
        tail_active = (self.tail_angle_speed_low_pass>self.BoutThresh)*1.0

        # Compute Running Average to compute bouts amplitude:
        onset,offset,duration = find_onset_offset_numpy(tail_active==1)
        # Refine Segmentation:
        onset, offset, duration, inter_bouts = refine_segmentation_bouts(onset, offset, Min_Duration, Min_IBI)
        onset, offset = onset.tolist(),offset.tolist()
        self.onset = onset
        self.offset = offset
        self.tail_active = np.zeros_like(tail_active)
        for on_,off_ in zip(onset,offset):
            self.tail_active[on_:off_]=1
        #self.initial_tail_active = tail_active
        return self.onset,self.offset,self.tail_active

    def extract_half_beat(self,Half_BCFilt = 150, stdThres = 5,MinSizeBlob=500,Margin=10):

        peaks_pos = []
        peaks_neg = []

        for i,(on_,off_) in enumerate(zip(self.onset,self.offset)):
            on_ = on_ - Margin
            off_ = off_ + Margin
            bout_slice = self.tail_angle_smooth[on_:off_,2:self.Reference_tail_segment+1]
            half_beat_pos,half_beat_neg,binary_image = find_half_beat(bout_slice,Half_BCFilt = Half_BCFilt, stdThres = stdThres,MinSizeBlob=MinSizeBlob)
            
            peaks_pos = peaks_pos + (half_beat_pos+on_).tolist()
            peaks_neg = peaks_neg + (half_beat_neg+on_).tolist()

        peaks_pos = np.array(peaks_pos)
        peaks_neg = np.array(peaks_neg)

        self.half_beat_pos = peaks_pos
        self.half_beat_neg = peaks_neg

        return peaks_pos,peaks_neg
    
    def refine_bouts(self,MaxIBeatI = 50):
        # Easier way : redefined tail active as a series of continuous bouts:
        all_peaks = np.concatenate((self.half_beat_pos,self.half_beat_neg))
        all_peaks = np.sort(all_peaks)
        tail_active = np.zeros_like(self.tail_active)
        for p1,p2 in zip(all_peaks[0:-1],all_peaks[1:]):
            if (p2-p1)<MaxIBeatI:
                tail_active[p1:p2] = 1
            else:
                tail_active[p1-5:p1+5] = 1
        
        # Recompute onset and offset:
        id = np.where(np.diff(tail_active)==1)[0]
        self.onset = id
        id = np.where(np.diff(tail_active)==-1)[0]
        self.offset = id
        self.tail_active = tail_active
        return self.onset,self.offset,self.tail_active
