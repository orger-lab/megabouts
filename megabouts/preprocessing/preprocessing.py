from dataclasses import dataclass,field
import numpy as np
import pandas as pd
from megabouts.preprocessing.smoothing import one_euro_filter,clean_using_pca,tail_imputing
from megabouts.preprocessing.trajectory import compute_kinematic_activity,compute_speed
from megabouts.utils.utils_downsampling import convert_ms_to_frames
from megabouts.utils.utils import robust_diff
from megabouts.preprocessing.baseline import compute_baseline
import scipy.signal as signal
from scipy.signal import savgol_filter



@dataclass(frozen=True)#,kw_only=True)
class Preprocessed_Traj():
    x: np.ndarray
    y: np.ndarray
    body_angle: np.ndarray
    axial_speed: np.ndarray
    lateral_speed: np.ndarray
    yaw_speed: np.ndarray
    kinematic_activity: np.ndarray


def preprocess_traj(*,x,y,body_angle,fps=700,fc_min=20,beta=1,robust_diff=15,lag=60):
    """ Smooth trajectory using 1€ filter and compute derivative

    Args:
        x (np.ndarray): x position in mm
        y (np.ndarray):  y position in mm
        body_angle (np.ndarray): yaw angle in radian
        fps (int): frame rate 
        fc_min (float): minimum cuoff frequency in Hz
        beta (float): cutoff slope
    Returns:
        Preprocessed_Traj: instance of class preprocessed_traj
    """ 
    smooth_func = lambda x : one_euro_filter(x,fc_min,beta,fps)
    x,y,body_angle  = map(smooth_func,[x,y,body_angle])

    axial_speed,lateral_speed,yaw_speed = compute_speed(x,y,body_angle,fps,n_diff=robust_diff)
    kinematic_activity = compute_kinematic_activity(axial_speed,lateral_speed,yaw_speed,lag=lag,fps=fps)

    preprocessed_traj = Preprocessed_Traj(x=x,
                                            y=y,
                                            body_angle=body_angle,
                                            axial_speed=axial_speed,
                                            lateral_speed=lateral_speed,
                                            yaw_speed=yaw_speed,
                                            kinematic_activity=kinematic_activity)
    return preprocessed_traj


def preprocess_tail(*,tail_angle,limit_na=5,num_pcs=4,savgol_window=11,
                    baseline_method,baseline_params):
    """Interpolate Nan in tail_angle and remove noise using PCA

    Args:
        tail_angle (np.ndarray): input tail angle
        limit_na (int, optional): if more frames are missing we don't interpolate. Defaults to 5.
        num_pcs (int, optional): number of principal components we keep. Defaults to 4.
        savgol_window (int, optional): odd windows size used to fit a second order polynom. Defaults to 11. -1 avoid smoothing

    Returns:
        np.ndarray: filtered tail angle
    """     
    
    tail_angle_interp = tail_imputing(tail_angle)
    
    # Interpolate NaN timestep:
    tail_angle_clean = np.zeros_like(tail_angle_interp)
    for s in range(tail_angle_interp.shape[1]):
        ds = pd.Series(tail_angle_interp[:,s])
        ds.interpolate(method='nearest',limit=limit_na,inplace=True)
        tail_angle_clean[:,s] = ds.values

    # Set to 0 for long sequence of nan:
    tail_angle_clean[np.isnan(tail_angle_clean)]=0

    # Use PCA for Cleaning 
    tail_angle_clean = clean_using_pca(tail_angle_clean,num_pcs=num_pcs)
    
    # Use Savgol filter:
    smooth_tail_angle=np.copy(tail_angle_clean)
    if savgol_window!=-1:
        for n in range(tail_angle_clean.shape[1]):
            smooth_tail_angle[:,n]=signal.savgol_filter(tail_angle_clean[:,n], savgol_window, 2, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

    # Remove baseline:
    baseline = np.zeros_like(smooth_tail_angle)
    for s in range(smooth_tail_angle.shape[1]):
        baseline[:,s] = compute_baseline(smooth_tail_angle[:,s],baseline_method,baseline_params)

    return smooth_tail_angle,baseline


def compute_tail_speed(*,tail_angle,fps,tail_speed_filter,tail_speed_boxcar_filter):
    """ Smooth trajectory using 1€ filter and compute derivative

    Args:
        tail_angle (np.ndarray): input tail angle
        fps (int): frame rate 
        tail_speed_filter (int): length to compute robust diff, should be odd
        tail_speed_boxcar_filter (int): size of boxcar filter to smooth half-beat oscillation
    Returns:
        np.ndarray: smooth_tail_speed
    """ 
    tail_angle_speed = np.zeros_like(tail_angle)
    for i in range(tail_angle.shape[1]):
        tail_angle_speed[:,i] = robust_diff(tail_angle[:,i],dt=1/fps,filter_length=tail_speed_filter)
    # Sum the angle differences down the length of the tail to give prominence to regions of continuous curvature in one direction
    cumul_filtered_speed=np.sum(np.abs(tail_angle_speed),axis=1)
    # Sum the absolute value of this accumulated curvature so bends in both directions are considered
    smooth_tail_speed= signal.convolve(
            cumul_filtered_speed,
            1/tail_speed_boxcar_filter*signal.boxcar(M=tail_speed_boxcar_filter,sym=False),
            mode='same')
    
    return smooth_tail_speed