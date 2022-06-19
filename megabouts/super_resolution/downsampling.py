import numpy as np
from scipy.interpolate import interp1d


def convert_duration(N_frame,original_fps,new_fps):

    return int(np.ceil(new_fps/original_fps*N_frame))



def create_downsampling_function(new_fps,duration_original=140,original_fps=700,kind='linear'):
    
    N_downsampled = convert_duration(duration_original,original_fps,new_fps)
    
    t = np.linspace(0,1000/original_fps*duration_original,duration_original,endpoint=False)

    tnew = np.linspace(0,1000/original_fps*duration_original,N_downsampled,endpoint=False)


    def downsampling_f(x,axis=0):
            
        # Make the interpolator function.
        func = interp1d(t, x, kind=kind,axis=axis)
        xnew = func(tnew)

        return xnew


    return downsampling_f, N_downsampled,t,tnew