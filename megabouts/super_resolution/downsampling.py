import numpy as np
from scipy.interpolate import interp1d


def convert_frame_duration(n_frames_origin:int,fps_origin:int,fps_new:int)->int:
    """_summary_

    Args:
        n_frames (int): number of frames in input times series
        fps_origin (int): fps of input times series
        fps_new (int): target fps

    Returns:
        int: number of frames corresponding for target fps
    """    

    return int(np.ceil(fps_new/fps_origin*n_frames_origin))

#TODO: REPLACE CONVERT DURATION USE BY CONVERT MS TO FRAMES WHEN POSSIBLE

def convert_ms_to_frames(fps:int,duration:float)->int:
    """Convert duration in ms to number of frames

    Args:
        fps (int): frame rate 
        duration (float): duration in ms

    Returns:
        int: number of frames correponding to duration
    """    
    n_frames = int(np.ceil(duration*fps/1000))
    return n_frames


def create_downsampling_function(fps_new:int,n_frames_origin:int,fps_origin=700,kind='linear'):
    """Generate function that will downsample according to fps_new

    Args:
        fps_new (int): target fps
        n_frames_origin (int): number of frames at the original fps
        fps_origin (int, optional): Defaults to 700.
        kind (str,optional): Defauts to linear, The string can be  'nearest', 'slinear', 'quadratic', 'cubic'...

    Returns:
        _type_: downsampling function that downsample a np.ndarray along a given axis
    """    
    
    n_frames_new = convert_frame_duration(n_frames_origin,fps_origin,fps_new)
    t = np.linspace(0,1000/fps_origin*n_frames_origin,n_frames_origin,endpoint=False)
    tnew = np.linspace(0,1000/fps_origin*n_frames_origin,n_frames_new,endpoint=False)

    def downsampling_f(x,axis=0):
            
        # Make the interpolator function.
        func = interp1d(t, x, kind=kind,axis=axis)
        xnew = func(tnew)

        return xnew

    return downsampling_f, n_frames_new,t,tnew