import os
import json

# Data Wrangling
import h5py
import numpy as np
import pandas as pd

# Bouts
from utils.utils_bouts import NameCat

# Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors

from pipeline.full_tracking import PipelineFullTracking
from pipeline.cfg import ConfigTrajPreprocess,ConfigTailPreprocess,ConfigSparseCoding,ConfigTailSegmentationClassification

from utils.utils_bouts import compute_bout_cat_ts
from utils.utils_plot import display_trajectory
from utils.utils import compute_outer_circle
from utils import smallestenclosingcircle as smallestenclosingcircle
    
from classification.benchmark_w_matlab import compute_bout_cat_matlab
from scipy import stats

##### LOAD PIPELINE #####

Dict = np.load('./sparse_coding/3atomsDictTu.npy')


cfg_tail_preprocess = ConfigTailPreprocess(fps=700,num_pcs=4,limit_na_ms=100,
                                           baseline_method='slow',
                                           baseline_params={'fps':700})

cfg_traj_preprocess = ConfigTrajPreprocess(fps=700,freq_cutoff_min=10,beta=4)


cfg_sparse_coding = ConfigSparseCoding(fps=700,Dict=Dict,lmbda=0.01,gamma=0.3,mu=0.3,window_inhib_ms=85)

cfg_segment_classify = ConfigTailSegmentationClassification(fps=700,
                                                            min_code_height=1,min_spike_dist_ms=200,
                                                            margin_before_peak_ms=32,
                                                            bout_duration_ms=200,
                                                            augment_max_delay_ms=18,
                                                            augment_step_delay_ms=2,
                                                            feature_weight= np.array([1.6]*7+[0.5,0.4,1]),
                                                            N_kNN=10)

pipeline = PipelineFullTracking(cfg_tail_preprocess,
                                cfg_traj_preprocess,
                                cfg_sparse_coding,
                                cfg_segment_classify,
                                load_training=True)

##### COLLECT ALL FISH #####


folder='D://ZebrafishMatlabCSV//'
folder_result = 'G://Megabouts_dataset//Full_Tracking_Pipeline//'
file_input_list=[]
file_result_list = []
for file in os.listdir(folder):
    if file.endswith(".csv"):
        file_input_list.append(os.path.join(folder, file))
        file_result_list.append(os.path.join(folder_result, file[:-7])+'.pickle')


##### RUN PIPELINE OVER ALL FISH #####

for filename,filename_result in zip(file_input_list[:],file_result_list[:]):

    df = pd.read_csv(filename)
    # Load Matlab pipeline results:
    onset_mat,offset_mat,bout_cat_matlab,bout_cat_ts_matlab,bout_cat_ts_signed_matlab = compute_bout_cat_matlab(df)
    matlab_results = {'onset':onset_mat,
                     'offset':offset_mat,
                     'bout_cat':bout_cat_matlab,
                     'bout_cat_ts':bout_cat_ts_matlab,
                     'bout_cat_ts_signed':bout_cat_ts_signed_matlab}

    x = df['x_blob'].values
    y = df['y_blob'].values
    body_angle = df['body_angle'].values
    body_angle = np.arctan2(np.sin(body_angle),np.cos(body_angle))
    body_angle = np.unwrap(body_angle)
    # Center trajectory:
    circle = compute_outer_circle(x,y)
    x,y = x-circle[0],y-circle[1]

    NumSegment = sum(['tail_angle' in df.columns[i] for i in range(len(df.columns))])
    relative_tail_angle = df[['tail_angle_'+str(i).zfill(2) for i in range(1,NumSegment+1)]]
    relative_tail_angle = relative_tail_angle.values
    tail_angle_init=np.cumsum(relative_tail_angle,1)
    tail_angle = np.copy(tail_angle_init)
    tail_angle[tail_angle<-10]=np.nan # Set Nan when tail angle out of range
    
    tracking_data,clean_traj,baseline,tail_angle_detrend,z,segments,tail_and_traj_array,bout_category,id_nearest_template = pipeline.run(tail_angle,x,y,body_angle)
    
    bout_cat_ts,bout_cat_ts_signed = compute_bout_cat_ts(segments.onset,
                                                         segments.offset,
                                                         bout_category,
                                                         tail_angle.shape[0])
    
    pipeline_results = {'tracking_data':tracking_data,
                        'clean_traj':clean_traj,
                        'baseline':baseline,
                        'tail_angle_detrend':tail_angle_detrend,
                        'z':z,
                        'segments':segments,
                        'tail_and_traj_array':tail_and_traj_array,
                        'bout_cat':bout_category}
    
    results = {'matlab':matlab_results,'pipeline':pipeline_results}
    
    import pickle
    # save dictionary to pickle file
    with open(filename_result, 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)