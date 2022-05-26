import numpy as np
import scipy.signal as signal
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import pandas as pd



def clean_using_pca(X,num_pcs=4):
    # X Should be T,NumSegments
    T=X.shape[0]
    NumSeg=X.shape[1]
    pca = PCA(n_components=num_pcs)
    pca.fit(X)
    low_D = pca.transform(X)
    X_clean = pca.inverse_transform(low_D)
    return X_clean


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


def create_preprocess(limit_na=5,num_pcs=4):
    
    def preprocess(tail_angle) :
        # Interpolate NaN:
        for s in range(tail_angle.shape[1]):
            ds = pd.Series(tail_angle[:,s])
            ds.interpolate(method='nearest',limit=limit_na)
            tail_angle[:,s] = ds.values

        # Set to 0 for long sequence of nan:
        tail_angle[np.isnan(tail_angle)]=0

        # Use PCA for Cleaning (Could use DMD for better results)
        tail_angle = clean_using_pca(tail_angle,num_pcs=num_pcs)
        return tail_angle

    return preprocess

'''
Do we need this since I didn't account for 
def compute_smooth_tail_angle(relative_tail_angle,thresh_error):
    tail_angle = relative_tail_angle
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
'''


