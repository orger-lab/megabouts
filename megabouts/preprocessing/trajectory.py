import numpy as np
from scipy.ndimage.interpolation import shift
from utils.utils import robust_diff
from preprocessing.smoothing import one_euro_filter

def compute_speed(x,y,body_angle,fps,n_diff=45):
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

def compute_mobility(axial_speed,lateral_speed,yaw_speed,lag = 140,fps=700):
    
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

    mobility = np.linalg.norm(displacement,axis=1)

    return mobility
