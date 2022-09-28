import numpy as np
import pandas as pd

import os
import json
import pickle

from tracking_data.dataset import Dataset_TailTracking,Dataset_CentroidTracking
from pipeline.cfg import ConfigTrajPreprocess,ConfigTrajSegmentationClassification
from pipeline.centroid_tracking import  PipelineCentroidTracking

from utils.utils_plot import display_trajectory
from utils.utils import compute_outer_circle
from utils import smallestenclosingcircle as smallestenclosingcircle
from super_resolution.downsampling import convert_frame_duration,create_downsampling_function

from utils.utils_bouts import NameCat

# Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors


from utils.utils_bouts import compute_bout_cat_ts
from classification.benchmark_w_matlab import compute_bout_cat_matlab
from scipy import stats

fps_list = np.linspace(30,700,30)
fps_list = 10*np.round(fps_list/10).astype('int')

peak_prominence_list = np.linspace(0.1,1,30)



folder='D://ZebrafishMatlabCSV//'
folder_fulltrack = 'G://Megabouts_dataset//Full_Tracking_Pipeline//'
folder_result = 'G://Megabouts_dataset//Super_Resolution_Pipeline//'
file_input_list=[]
file_fulltrack_list = []
file_result_list = []

for file in os.listdir(folder):
    if file.endswith(".csv"):
        file_input_list.append(os.path.join(folder, file))
        file_fulltrack_list.append(os.path.join(folder_fulltrack, file[:-7])+'.pickle')
        file_result_list.append(os.path.join(folder_result, file[:-7]))
        
id_rnd = np.random.RandomState(seed=42).permutation(np.arange(0,len(file_input_list)))[0:20]

file_input_list = np.array(file_input_list)[id_rnd]
file_fulltrack_list = np.array(file_fulltrack_list)[id_rnd]
file_result_list = np.array(file_result_list)[id_rnd]
        

for i_param,(fps_new,peak_pro) in enumerate(zip(fps_list[24:],peak_prominence_list[24:])):
    print(fps_new)
    
    # Create Pipeline:
    cfg_preprocess = ConfigTrajPreprocess(fps=fps_new,freq_cutoff_min=10,beta=4)
    cfg_segment_classify = ConfigTrajSegmentationClassification(fps=fps_new,peak_prominence=peak_pro,margin_before_peak_ms=34,augment_max_delay_ms=34,augment_step_delay_ms=4,N_kNN=10)
    pipeline = PipelineCentroidTracking(cfg_preprocess,cfg_segment_classify,load_training=True)
    
    
    for filename,filename_fulltrack,filename_result in zip(file_input_list,file_fulltrack_list,file_result_list):
        
        # Load tracking data:
        df = pd.read_csv(filename)

        x = df['x_blob'].values
        y = df['y_blob'].values
        body_angle = df['body_angle'].values
        body_angle = np.arctan2(np.sin(body_angle),np.cos(body_angle))
        body_angle = np.unwrap(body_angle)
        # Center trajectory:
        circle = compute_outer_circle(x,y)
        x,y = x-circle[0],y-circle[1]
                        
        # Downsample:
        #downsampling_f, Duration_after_Downsampling,original_t,new_t = create_downsampling_function(fps_new=fps_new,n_frames_origin=len(x),fps_origin=700)
        downsampling_f, Duration_after_Downsampling,t,tnew = create_downsampling_function(fps_new=fps_new,fps_origin=700,duration_ms=len(x)*1000/700)
        
        x_sub,y_sub,body_angle_sub  = map(lambda x : downsampling_f(x,axis=0),[x,y,body_angle])
        
        # Apply pipeline:
        tracking_data,clean_traj,segments,segment_refined,traj_array,bout_category,onset_delay,id_nearest_template = pipeline.run(x_sub,y_sub,body_angle_sub)
        
        bout_cat_ts,bout_cat_ts_signed = compute_bout_cat_ts(segment_refined.onset,
                                                            segment_refined.offset,
                                                            bout_category,
                                                            body_angle_sub.shape[0])
        
        # Save data:
        
        pipeline_results = {'clean_traj':clean_traj,
                            'segments':segment_refined,
                            'traj_array':traj_array,
                            'bout_cat':bout_category}
                
        # save dictionary to pickle file
        
        filename_results_fps = f"{filename_result}_fps{fps_new:03d}.pickle"

        with open(filename_results_fps, 'wb') as file:
            pickle.dump(pipeline_results, file, protocol=pickle.HIGHEST_PROTOCOL)

