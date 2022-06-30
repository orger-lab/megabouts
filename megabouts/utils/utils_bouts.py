import numpy as np


NameCatSym = ['approach_swim','slow1','slow2','slow_capture_swim','fast_capture_swim','burst_swim','J_turn','high_angle_turn','routine_turn','spot_avoidance_turn','O_bend','long_latency_C_start','C_start']
NameCat = [s+'+' for s in NameCatSym] + [s+'-' for s in NameCatSym]#[::-1] 
    


def compute_bout_cat_ts(onset,offset,bout_cat,T):
    # COMPUTE BOUT CAT MATRIX:
    bout_cat_ts = -1+np.zeros(T)
    for on_,off_,b in zip(onset,offset,bout_cat):
        bout_cat_ts[on_:off_] = b%13

    bout_cat_ts_signed = bout_cat_ts[:,np.newaxis]
    bout_cat_ts_signed = np.concatenate((bout_cat_ts_signed,bout_cat_ts_signed),axis=1)
    bout_cat_ts_signed[bout_cat_ts_signed[:,0]>12,0] = -1
    bout_cat_ts_signed[bout_cat_ts_signed[:,1]<13,1] = -1
    bout_cat_ts_signed[bout_cat_ts_signed[:,1]>-1,1] = bout_cat_ts_signed[bout_cat_ts_signed[:,1]>-1,1]%13

    return bout_cat_ts,bout_cat_ts_signed

