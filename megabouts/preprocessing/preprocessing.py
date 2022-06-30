from dataclasses import dataclass,field
import numpy as np
from preprocessing.smoothing import one_euro_filter
from preprocessing.trajectory import compute_mobility,compute_speed
from super_resolution.downsampling import convert_ms_to_frames

@dataclass(frozen=True)#,kw_only=True)
class Preprocessed_Traj():
    x: np.ndarray
    y: np.ndarray
    body_angle: np.ndarray
    axial_speed: np.ndarray
    lateral_speed: np.ndarray
    yaw_speed: np.ndarray
    mobility: np.ndarray


def create_preprocess_traj(fps=700,fc_min=20,beta=1,robust_diff=15,lag=60):
    
    # Convert Duration to number of frames:

    def preprocess(x,y,body_angle) :
        """ Smooth trajectory using 1€ filter and compute derivative

        Args:
            x (np.ndarray): x position in mm
            y (np.ndarray):  y position in mm
            body_angle (np.ndarray): yaw angle in radian

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

    return preprocess


def create_preprocess_tail(limit_na=5,num_pcs=4):
    
    def preprocess(tail_angle) :
        """ Interpolate Nan in tail_angle and remove noise using PCA

        Args:
            tail_angle (np.ndarray): input tail angle

        Returns:
            np.ndarray: filtered tail angle
        """ 
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

