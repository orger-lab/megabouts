from dataclasses import dataclass,field
import numpy as np
import pandas as pd
from preprocessing.smoothing import one_euro_filter,clean_using_pca
from preprocessing.trajectory import compute_mobility,compute_speed
from super_resolution.downsampling import convert_ms_to_frames
from preprocessing.baseline import compute_baseline


@dataclass(frozen=True)#,kw_only=True)
class Preprocessed_Traj():
    x: np.ndarray
    y: np.ndarray
    body_angle: np.ndarray
    axial_speed: np.ndarray
    lateral_speed: np.ndarray
    yaw_speed: np.ndarray
    mobility: np.ndarray


def preprocess_traj(*,x,y,body_angle,fps=700,fc_min=20,beta=1,robust_diff=15,lag=60):
    """ Smooth trajectory using 1€ filter and compute derivative

    Args:
        x (np.ndarray): x position in mm
        y (np.ndarray):  y position in mm
        body_angle (np.ndarray): yaw angle in radian
        fps (int): frame rate def
        fc_min (float): minimum cuoff frequency in Hz
        beta (float): cutoff slope
    Returns:
        Preprocessed_Traj: instance of class preprocessed_traj
    """ 
    smooth_func = lambda x : one_euro_filter(x,fc_min,beta,fps)
    x,y,body_angle  = map(smooth_func,[x,y,body_angle])

    axial_speed,lateral_speed,yaw_speed = compute_speed(x,y,body_angle,fps,n_diff=robust_diff)
    mobility = compute_mobility(axial_speed,lateral_speed,yaw_speed,lag=lag,fps=fps)

    preprocessed_traj = Preprocessed_Traj(x=x,
                                            y=y,
                                            body_angle=body_angle,
                                            axial_speed=axial_speed,
                                            lateral_speed=lateral_speed,
                                            yaw_speed=yaw_speed,
                                            mobility=mobility)
    return preprocessed_traj


def preprocess_tail(*,tail_angle,limit_na=5,num_pcs=4,baseline_method,baseline_params):
    """Interpolate Nan in tail_angle and remove noise using PCA

    Args:
        tail_angle (np.ndarray): input tail angle
        limit_na (int, optional): _description_. Defaults to 5.
        num_pcs (int, optional): _description_. Defaults to 4.

    Returns:
        np.ndarray: filtered tail angle
    """ 
    # Interpolate NaN:
    tail_angle_clean = np.copy(tail_angle)
    for s in range(tail_angle.shape[1]):
        ds = pd.Series(tail_angle[:,s])
        ds.interpolate(method='nearest',limit=limit_na,inplace=True)
        tail_angle_clean[:,s] = ds.values

    # Set to 0 for long sequence of nan:
    tail_angle_clean[np.isnan(tail_angle_clean)]=0

    # Use PCA for Cleaning (Could use DMD for better results)
    tail_angle_clean = clean_using_pca(tail_angle_clean,num_pcs=num_pcs)
    
    # Remove baseline:
    baseline = np.zeros_like(tail_angle_clean)
    for s in range(tail_angle_clean.shape[1]):
        baseline[:,s] = compute_baseline(tail_angle_clean[:,s],baseline_method,baseline_params)

    return tail_angle_clean,baseline


