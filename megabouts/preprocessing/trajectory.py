import numpy as np
from scipy.ndimage.interpolation import shift
from megabouts.utils.utils import robust_diff
from megabouts.preprocessing.smoothing import one_euro_filter

def compute_speed(x,y,body_angle,fps,n_diff=45):
    """
    Compute the axial, lateral, and yaw speed of a body given its position, body angle, and FPS.

    Parameters:
    - x: 1D numpy array of x-coordinates of the body's position.
    - y: 1D numpy array of y-coordinates of the body's position.
    - body_angle: 1D numpy array of the body's angle.
    - fps: float, frames per second at which the body's position and angle were recorded.
    - n_diff: int, number of frames to use in the robust difference computation.

    Returns:
    - axial_speed: 1D numpy array of the body's axial speed.
    - lateral_speed: 1D numpy array of the body's lateral speed.
    - yaw_speed: 1D numpy array of the body's yaw speed.
    
    """
    body_vector = np.array([np.cos(body_angle),np.sin(body_angle)])[:,:-1]
    position_change = np.zeros_like(body_vector)
    position_change[0,:] = robust_diff(x,dt=1/fps, filter_length=n_diff)[:-1]
    position_change[1,:] = robust_diff(y,dt=1/fps, filter_length=n_diff)[:-1]
    angle = np.pi/2
    rotMat = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    body_vector_orth = np.dot(rotMat,body_vector)

    axial_speed = np.einsum('ij,ij->j',body_vector,position_change)
    lateral_speed = np.einsum('ij,ij->j',body_vector_orth,position_change)
    yaw_speed = robust_diff(np.unwrap(body_angle),dt=1/fps, filter_length=n_diff)[:-1]
    axial_speed=np.concatenate((np.array([np.nan]),axial_speed))
    lateral_speed=np.concatenate((np.array([np.nan]),lateral_speed))
    yaw_speed=np.concatenate((np.array([np.nan]),yaw_speed))
    speed_amplitude = np.sqrt(np.power(axial_speed,2)+np.power(lateral_speed,2))
    
    return axial_speed,lateral_speed,yaw_speed

def compute_kinematic_activity(axial_speed,lateral_speed,yaw_speed,lag = 140,fps=700):
    """
    Compute the kinematic_activity of a body given its axial, lateral, and yaw speed.

    Parameters:
    - axial_speed: 1D numpy array of the body's axial speed.
    - lateral_speed: 1D numpy array of the body's lateral speed.
    - yaw_speed: 1D numpy array of the body's yaw speed.
    - lag: int, number of frames to use in the kinematic_activity calculation.
    - fps: float, frames per second at which the body's speed was recorded.

    Returns:
    - kinematic_activity: 1D numpy array of the body's kinematic_activity.
    """
    
    traj_speed = np.vstack((axial_speed,lateral_speed,yaw_speed)).T
    traj_speed[np.isnan(traj_speed)]=0
    traj_cumul = np.cumsum(np.abs(traj_speed),axis=0)

    dt = 1000/fps*lag
    displacement = np.zeros_like(traj_cumul)
    for i in range(3):
        tmp= np.copy(traj_cumul[:,i])
        tmp_future_lag=shift(tmp,-lag, cval=0)
        displacement[:,i]=(tmp_future_lag-tmp)/dt

    displacement[:lag,:] = 0
    displacement[-lag:,:] = 0

    kinematic_activity = np.linalg.norm(displacement,axis=1)

    return kinematic_activity
