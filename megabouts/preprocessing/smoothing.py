import numpy as np
import scipy.signal as signal
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

def clean_using_pca(X:np.ndarray,num_pcs=4)->np.ndarray:
    """Apply PCA autoencoding to clean up a multidimensional time series

    Args:
        X (np.ndarray): tail angle, should be of size (T,num_features)
        num_pcs (int, optional): Cutoff on number of principal components. Defaults to 4.

    Returns:
        np.ndarray: return X_hat
    """
    # X Should be T,NumSegments
    T=X.shape[0]
    num_features=X.shape[1]
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
    x_smooth[0] = x[0]
    
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




