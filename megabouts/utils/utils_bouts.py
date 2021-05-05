from scipy.ndimage.filters import maximum_filter1d 
import numpy as np
import scipy.signal as signal

from skimage.measure import label, regionprops, regionprops_table


from sklearn.decomposition import PCA

def max_filter1d_valid(a, W):
    hW = (W-1)//2 # Half window size
    return maximum_filter1d(a,size=W,mode='reflect')#[hW:-hW]


# Fast Derivative Computation:
from scipy.special import binom

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

def clean_using_pca(X,num_pcs=4):
    
    # X Should be NumBouts,T,NumSegments
    T=X.shape[0]
    NumSeg=X.shape[1]

    pca = PCA(n_components=num_pcs)
    pca.fit(X)
    low_D = pca.transform(X)

    X_clean = pca.inverse_transform(low_D)
    
    return X_clean
    
def find_half_beat(tail,ThreshSizeBeat=40):
    speed = np.zeros_like(tail)
    speed_pos = np.zeros_like(tail)
    speed_neg = np.zeros_like(tail)

    for s in range(tail.shape[1]):
        speed[:,s] = derivative_n2(tail[:,s], dt=1/700, filter_length=7)
        Thresh = np.std(speed[~np.isnan(speed[:,s]),s])
        speed_pos[speed[:,s]>Thresh/5,s]=1
        speed_neg[speed[:,s]<-Thresh/5,s]=1

    peak_loc_pos,peak_loc_neg = [],[]

    for img,peak_loc in zip([speed_pos,speed_neg],[peak_loc_pos,peak_loc_neg]):

        label_img = label(img.astype('int'))
        regions = regionprops(label_img)    
        for i,props in enumerate(regions):
            if props.area>ThreshSizeBeat:
                half_beat = label_img==(i+1)
                id_ = np.where(half_beat[:,-1]==1)[0]
                if len(id_)>0:
                    peak_loc.append(id_[-1]) # Last index <-> Decrease in speed near max
    
    return peak_loc_pos,peak_loc_neg,speed_pos,speed_neg
        
            

def compute_smooth_tail_angle(tail_angle):

    # Define Cumul Sum and Find Tracking Nan:
    error_flag=np.where(np.min(tail_angle,1)<-50)[0]
    tail_angle[tail_angle==-100]=0
    cumul_tail_angle=np.cumsum(tail_angle,1)
    num_errors=error_flag.shape[0]

    # For the bout detection we smooth the tail curvature to eliminate kinks due to tracking noise
    notrack=np.where(np.sum(cumul_tail_angle,1)==0)[0]
    print('Shape of No Track:')
    print(notrack.shape)
    smooth_cumul_tail_angle=np.copy(cumul_tail_angle)
    for n in range(1,cumul_tail_angle.shape[1]-1):
        smooth_cumul_tail_angle[:,n]=np.mean(cumul_tail_angle[:,n-1:n+2],1)

    for n in range(cumul_tail_angle.shape[1]):
        tmp=cumul_tail_angle[:,n]
        tmp=signal.savgol_filter(tmp, 11, 2, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        smooth_cumul_tail_angle[:,n]=tmp
        
    return cumul_tail_angle,smooth_cumul_tail_angle,notrack

def compute_tail_speed(smooth_cumul_tail_angle,notrack,NumSegments,BCFilt):
    
    # Compute difference in segment angles, because we want to detect tail movement
    tmp=np.diff(smooth_cumul_tail_angle,axis=0)
    z=np.zeros((1,tmp.shape[1]))
    speed_tail_angle=np.vstack((z,tmp))
    speed_tail_angle[notrack,:]=0
    
    # Interpolate on NoTrack
    if notrack.shape[0]>0:
        if notrack[-1]==smooth_cumul_tail_angle.shape[0]-1:
            speed_tail_angle[notrack[:-1]+1,:]=0
        else:
            speed_tail_angle[notrack+1,:]=0
    speed_tail_angle = speed_tail_angle[:,0:NumSegments];
    
    
    # Smooth the tail movement
    filtered_speed_tail_angle=np.zeros_like(speed_tail_angle)
    for i in range(speed_tail_angle.shape[1]):
        filtered_speed_tail_angle[:,i]= signal.convolve(
            speed_tail_angle[:,i],
            1/BCFilt*signal.boxcar(M=BCFilt,sym=False),
            mode='same')

    # Sum the angle differences down the length of the tail to give prominence to regions of continuous curvature in one direction
    cumul_filtered_speed=np.cumsum(filtered_speed_tail_angle,1)

    # Sum the absolute value of this accumulated curvature so bends in both directions are considered
    super_cumul = np.cumsum(np.abs(cumul_filtered_speed),1)
    smooth_tail_speed= signal.convolve(
            super_cumul[:,-1],
            1/BCFilt*signal.boxcar(M=BCFilt,sym=False),
            mode='same')

    return smooth_tail_speed,speed_tail_angle,super_cumul

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


def clean_using_pca(X,num_pcs=4):
    
    # X Should be NumBouts,T,NumSegments
    T=X.shape[0]
    NumSeg=X.shape[1]

    pca = PCA(n_components=num_pcs)
    pca.fit(X)
    low_D = pca.transform(X)

    X_clean = pca.inverse_transform(low_D)
    
    return X_clean