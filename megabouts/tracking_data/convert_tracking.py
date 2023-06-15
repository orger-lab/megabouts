from dataclasses import dataclass,field
import numpy as np
from scipy.interpolate import interp1d
from megabouts.utils.utils import compute_angle_between_vectors

def interpolate_tail_keypoint(tail_x,tail_y,n_segments=10):
    """
    Interpolates the tail keypoints to create a curve with a specified number of segments.

    Parameters:
        tail_x (numpy.ndarray): The x-coordinates of the tail keypoints. Shape: (T, n_segments_input).
        tail_y (numpy.ndarray): The y-coordinates of the tail keypoints. Shape: (T, n_segments_input).
        n_segments (int, optional): The number of segments to interpolate. Default: 10.

    Returns:
        numpy.ndarray: The interpolated x-coordinates of the tail keypoints. Shape: (T,n_segments+1).
        numpy.ndarray: The interpolated y-coordinates of the tail keypoints. Shape: (T,n_segments+1).
    """
    if n_segments<2:
        raise ValueError("there should be more than 3 keypoints.")     
                
    # Extrapolate to 10 segments:
    T, n_segments_init = tail_x.shape[0], tail_x.shape[1]
    tail_x_interp = np.zeros((T,n_segments+1))*np.nan
    tail_y_interp = np.zeros((T,n_segments+1))*np.nan
    for i in range(T):
        try:
            points = np.array([tail_x[i,:],tail_y[i,:]]).T  # a (nbre_points x nbre_dim) array
            is_nan = np.any(np.isnan(points))
            
            if is_nan==False:
                id_first_nan=points.shape[0]
                N_seg = n_segments+1
                
            else: 
                id_first_nan = np.where(np.any(np.isnan(points),axis=1))[0][0]
                N_seg = int(np.round(id_first_nan/n_segments_init*(n_segments+1)))
                
            alpha = np.linspace(0, 1, N_seg)
            # Linear length along the line:
            distance = np.cumsum( np.sqrt(np.sum( np.diff(points[:id_first_nan,:], axis=0)**2, axis=1 )) )
            distance = np.insert(distance, 0, 0)/distance[-1]
            if len(distance)>3:
                interpolator =  interp1d(distance, points[:id_first_nan,:], kind='cubic', axis=0)
            else:
                interpolator =  interp1d(distance, points[:id_first_nan,:], kind='linear', axis=0)

            curve = interpolator(alpha)
            
            tail_x_interp[i,:N_seg] = curve[:,0]
            tail_y_interp[i,:N_seg] = curve[:,1]
        except:
            pass
            
    return tail_x_interp,tail_y_interp

def compute_angles_from_keypoints(body_x, body_y, tail_x, tail_y):
    """
    Computes the tail angles and body angle based on keypoints.
    
    Args:
        body_x (ndarray): X-coordinates of the body keypoints.
        body_y (ndarray): Y-coordinates of the body keypoints.
        tail_x (ndarray): X-coordinates of the tail keypoints. 
        tail_y (ndarray): Y-coordinates of the tail keypoints. .
    Returns:
        tuple: A tuple containing the tail angles (tail_angle) and the body angle (body_angle).
    
    Raises:
        ValueError: If tail_x and tail_y do not have the expected shape or if the time axis is different.
    """
    N_keypoints = tail_x.shape[1]
    if tail_x.shape[1]!=tail_y.shape[1]:
        raise ValueError("tail_x and tail_y must have same dimensions")

    T = len(body_x)
    vector_tail_segment = np.concatenate((np.diff(tail_x, axis=1)[:, :, np.newaxis], np.diff(tail_y, axis=1)[:, :, np.newaxis]), axis=2)
    vector_tail_segment = np.swapaxes(vector_tail_segment, 1, 2)
    start_vector = np.vstack((tail_x[:, 0] - body_x, tail_y[:, 0] - body_y)).T
    body_vector = -start_vector
    body_angle = np.arctan2(body_vector[:, 1], body_vector[:, 0])
    body_angle[~np.isnan(body_angle)] = np.unwrap(body_angle[~np.isnan(body_angle)])
    relative_angle = np.zeros((T, N_keypoints-1))
    print(vector_tail_segment.shape)
    for i in range(N_keypoints-1):
        relative_angle[:, i] = compute_angle_between_vectors(start_vector, vector_tail_segment[:, :, i])
        start_vector = np.copy(vector_tail_segment[:, :, i])
    tail_angle = np.cumsum(relative_angle, 1)

    return tail_angle, body_angle


def convert_tail_angle_to_keypoints(body_x,body_y,body_angle,tail_angle,body_to_tail_mm=0.5,tail_to_tail_mm=0.32):
    
    T = tail_angle.shape[0]
    num_angle = tail_angle.shape[1]
    if ((len(body_x)!=T)or(len(body_x)!=T)or(len(body_angle)!=T)):
        raise ValueError("incompatible dimensions")

    num_segments = num_angle+1


    tail_x = np.zeros((T,num_segments))
    tail_y = np.zeros((T,num_segments))
    
    for i in range(T):
        x,y = body_x[i],body_y[i]
        head_pos = np.array([x,y])
        body_vect = np.array([np.cos(body_angle[i]),np.sin(body_angle[i])])
        
        swim_bladder = head_pos - body_vect*body_to_tail_mm
        
        tail_x[i,0] = swim_bladder[0]
        tail_y[i,0] = swim_bladder[1]
        tail_angle_abs = tail_angle[i,:]+(body_angle[i]+np.pi)
        tail_pos = np.copy(swim_bladder)
        for j in range(num_angle):
            tail_vect = np.array([np.cos(tail_angle_abs[j]),np.sin(tail_angle_abs[j])])
            tail_pos += tail_to_tail_mm*tail_vect
            tail_x[i,j+1] = tail_pos[0]
            tail_y[i,j+1] = tail_pos[1]
        
    return tail_x,tail_y    

def interpolate_tail_angle(tail_angle,n_segments=10):
    
    T = tail_angle.shape[0]
    N_seg = tail_angle.shape[1]
    # Convert to keypoints
    body_x,body_y,body_angle = np.zeros(T),np.zeros(T),np.zeros(T)
    tail_x,tail_y = convert_tail_angle_to_keypoints(body_x,body_y,body_angle,tail_angle,body_to_tail_mm=0.5,tail_size_mm=0.32)
    
    # Interpolate
    tail_x_interp,tail_y_interp = interpolate_tail_keypoint(tail_x,tail_y,n_segments=n_segments)
    
    # Compute tail angle
    tail_angle_interp, body_angle_interp = compute_angles_from_keypoints(body_x, body_y, tail_x_interp, tail_y_interp)
    
    return tail_angle_interp
    
