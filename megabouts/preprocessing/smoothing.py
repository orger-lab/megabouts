import numpy as np
import scipy.signal as signal
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def tail_imputing(tail_angle):
    """Use sklearn to Imput missing tail segments:

    Args:
        X (np.ndarray): tail angle, should be of size (T,num_features)
    Returns:
        np.ndarray: return X where missing segments are interpolated
    """

    # Find when tail is moving:
    tail_angle_train = np.copy(tail_angle)
    tail_angle_train = tail_angle_train[np.isnan(tail_angle[:,-1])==False,:]
    tail_angle_train = tail_angle_train[np.abs(tail_angle_train[:,-1])>0.5,:]
    if tail_angle_train.shape[0]>10000:
        tail_angle_train = tail_angle_train[:10000,:]
        
    # Train Imputer:
    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(tail_angle_train)

    # Interpolate missing segments using the rest of the tail:
    tail_angle_interp = np.copy(tail_angle)
    no_segment_tracked = np.all(np.isnan(tail_angle_interp),axis=1)
    tail_angle_interp[no_segment_tracked==False,:] = imp.transform(tail_angle[no_segment_tracked==False,:])
    
    return tail_angle_interp

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




