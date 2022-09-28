from scipy import stats
import numpy as np
from utils.utils import find_onset_offset_numpy
from utils.utils_bouts import compute_bout_cat_ts
from super_resolution.downsampling import create_downsampling_function

def compute_bout_cat_matlab(df,fps_new=700,fps_old=700):

    tail_active =  df.tail_active.values
    bout_cat_init = df.bout_sign_matlab.values
    
    downsampling_f, Duration_after_Downsampling,t,tnew = create_downsampling_function(fps_new=self.fps,fps_origin=fps_old,duration_ms=len(tail_active)*1000/700)
    #downsampling_f, Duration_after_Downsampling,t,tnew = create_downsampling_function(fps_new,n_frames_origin=len(tail_active),fps_origin=fps_old,kind='nearest')
    tail_active = downsampling_f(tail_active,axis = 0)
    bout_cat_init = downsampling_f(bout_cat_init,axis = 0)

    onset_mat,offset_mat,dur_mat = find_onset_offset_numpy(tail_active)
    bout_cat_matlab = []
    for on_,off_ in zip(onset_mat,offset_mat):
        bout_cat_matlab.append(bout_cat_init[on_+1])
    bout_cat_matlab = np.array(bout_cat_matlab)

    ArgBouts=np.array([10,6,8,0,1,2,4,12,7,11,3,9,5])
    bout_cat_matlab_ordered = np.zeros_like(bout_cat_matlab)
    for b_new,b_old in enumerate(ArgBouts):
        bout_cat_matlab_ordered[bout_cat_matlab==b_old]=b_new
    
    bout_cat_ts,bout_cat_ts_signed = compute_bout_cat_ts(onset_mat,offset_mat,bout_cat_matlab_ordered,tail_active.shape[0])


    return onset_mat,offset_mat,bout_cat_matlab_ordered,bout_cat_ts,bout_cat_ts_signed