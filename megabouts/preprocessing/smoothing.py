import numpy as np
import scipy.signal as signal
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import pandas as pd

def clean_using_pca(X:np.ndarray,num_pcs=4)->np.ndarray:
    """Apply PCA autoencoding to clean up the tail angle time series

    Args:
        X (np.ndarray): tail angle, should be of size (T,num_tail_segments)
        num_pcs (int, optional): Cutoff on number of principal components. Defaults to 4.

    Returns:
        np.ndarray: return X_hat
    """
    # X Should be T,NumSegments
    T=X.shape[0]
    num_tail_segments=X.shape[1]
    pca = PCA(n_components=num_pcs)
    pca.fit(X)
    low_D = pca.transform(X)
    X_hat = pca.inverse_transform(low_D)
    return X_hat

def one_euro_filter(x:np.ndarray,fc_min:float,beta:float,rate:int)->np.ndarray:
    """Apply 1€ filter over x

    Args:
        x (np.ndarray): input array to filter, size (n_frames,)
        fc_min (float): minimum cuoff frequency in Hz
        beta (float): cutoff slope
        rate (int): fps on x

    Returns:
        np.ndarray: filtered input
    """    
    n_frames = len(x)
    dx = 0
    x_smooth = np.zeros_like(x)

    fc = fc_min
    tau = 1/(2*np.pi*fc)
    te=1/rate
    alpha = 1/(1+tau/te)
    
    for i in range(1,n_frames):
                
        x_smooth[i] = alpha * x[i] + (1-alpha) * x_smooth[i-1]
        
        x_dot = (x_smooth[i]-x_smooth[i-1])*rate
        fc = fc_min + beta*np.abs(x_dot)
        tau = 1/(2*np.pi*fc)
        te=1/rate
        alpha = 1/(1+tau/te)

    return x_smooth


#TODO: Should this be here?
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



