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

from tracking_data.dataset import Dataset_TailTracking
from pipeline.tail_tracking import PipelineTailTracking
from pipeline.cfg import ConfigTailPreprocess,ConfigSparseCoding,ConfigTailSegmentationClassification


from utils.utils_bouts import compute_bout_cat_ts

from classification.benchmark_w_matlab import compute_bout_cat_matlab
from scipy import stats

##### LOAD PIPELINE #####

Dict = np.load('./sparse_coding/3atomsDictHR.npy')

cfg_preprocess = ConfigTailPreprocess(fps=700,num_pcs=3,limit_na_ms=100,
                                      baseline_method='whittaker',baseline_params={'fps':700,'lmbda':1e4})

cfg_sparse_coding = ConfigSparseCoding(fps=700,Dict=Dict,lmbda=0.01,gamma=0.15,mu=0.15,window_inhib_ms=85)

cfg_segment_classify = ConfigTailSegmentationClassification(fps=700,
                                                            min_code_height=1,min_spike_dist_ms=120,
                                                            margin_before_peak_ms=32,
                                                            bout_duration_ms=200,
                                                            augment_max_delay_ms=18,
                                                            augment_step_delay_ms=2,
                                                            feature_weight=np.ones(7),
                                                            N_kNN=10)


pipeline = PipelineTailTracking(cfg_preprocess,
                                cfg_sparse_coding,
                                cfg_segment_classify,
                                load_training=True)

##### COLLECT ALL FISH #####


folder='H://HeadRestrainedDataset//all_csv//'
folder_result = 'G://Megabouts_dataset//Head_Restrained_Pipeline//'
file_input_list=[]
file_result_list = []
for file in os.listdir(folder):
    if file.endswith(".csv"):
        file_input_list.append(os.path.join(folder, file))
        file_result_list.append(os.path.join(folder_result, file[:-7])+'.pickle')

##### RUN PIPELINE OVER ALL FISH #####

for filename,filename_result in zip(file_input_list,file_result_list):

    print(filename)
    df = pd.read_csv(filename)
    
    NumSegment = 16

    relative_tail_angle = df[['angle'+str(i) for i in range(NumSegment)]]
    relative_tail_angle = relative_tail_angle.values

    cumul_tail_angle=np.cumsum(relative_tail_angle,1)
    cumul_tail_angle[cumul_tail_angle<-10]=np.nan

    tracking_data = Dataset_TailTracking(fps=700,tail_angle=cumul_tail_angle)

    tail_angle_detrend,tail_angle_clean,baseline,z,tail_angle_hat,decomposition,segments,segment_refined,tail_array,bout_category,id_nearest_template = pipeline.run(tracking_data.tail_angle)
    
    bout_cat_ts,bout_cat_ts_signed = compute_bout_cat_ts(segments.onset,
                                                         segments.offset,
                                                         bout_category,
                                                         cumul_tail_angle.shape[0])
    
    pipeline_results = {'tracking_data':tracking_data,
                        'baseline':baseline,
                        'tail_angle_detrend':tail_angle_detrend,
                        'z':z,
                        'segments':segments,
                        'tail_array':tail_array,
                        'bout_cat':bout_category}
    
    results = {'pipeline':pipeline_results}
    
    import pickle
    # save dictionary to pickle file
    with open(filename_result, 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)