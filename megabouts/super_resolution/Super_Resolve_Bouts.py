import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


from Load_Ethogram import ListExpe,FishTable,add_hierarchical_level
#from .Load_Ethogram import Fish
from Fish import Fish
from TrajectoryPreprocessing import TrajectoryPreprocessing
from ShaderPreprocessing import ShaderPreprocessing

from Stimuli import EthogramStimuliProcessing
from scipy.signal import savgol_filter
from utils import diff_but_better
from utils import one_euro_filter
from utils import align_bout_peaks
from utils import find_onset_offset_numpy
from utils import mexican_hat_tail_speed,estimate_speed_threshold,max_filter1d_valid
#from utils import find_nearest_bouts_parallel

from scipy.spatial import distance
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsClassifier

import torch


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


###################################################################
################# DEFINE DATASET FOR KNN ##########################
###################################################################

arr = np.load('CleanBalancedBoutDataset.npz')
ref_bouts = arr['bouts']
label = arr['label']
NameCat = ['approach_swim','slow1','slow2','slow_capture_swim','fast_capture_swim','burst_swim','J_turn','high_angle_turn','routine_turn','spot_avoidance_turn','O_bend','long_latency_C_start','C_start']


N_Sample = 1400
balanced_bouts = np.zeros((0,ref_bouts.shape[1]))
balanced_label = []
for b in range(13):
    id_ = np.where(label==b)[0]
    print(len(id_))
    id_balanced = np.random.permutation(id_)[:N_Sample]
    balanced_bouts = np.vstack((balanced_bouts,ref_bouts[id_balanced,:]))
    balanced_label = balanced_label + [b]*N_Sample

balanced_label = np.array(balanced_label)

ref_bouts = balanced_bouts 
label = balanced_label 

ref_bouts = ref_bouts[(label!=3)&(label!=4)] # Remove slow and fast capture swim
label = label[(label!=3)&(label!=4)]  # Remove slow and fast capture swim

Duration_after_Downsampling = 80
ref_traj = np.zeros((len(label),3,Duration_after_Downsampling))

for i in range(len(label)):
    tmp = ref_bouts[i,:]
    tmp = np.reshape(tmp,(10,140))
    tmp = tmp[7:,:]  # Only use the trajectory
    for j in range(3):
        ref_traj[i,j,:]= zoom(tmp[j,:],400/700,order=3)

ref_traj[:,0,:],ref_traj[:,1,:],ref_traj[:,2,:] = ref_traj[:,0,:]*5,ref_traj[:,1,:]*5,ref_traj[:,2,:]/2 # Rescaling to make x,y in mm and angle in radian

delays = np.arange(10,60,3) # List of shift we are going to use:
ref_traj_rolled = np.zeros((len(label)*len(delays),3,80))
ref_label_rolled = np.nan*np.ones(len(label)*len(delays))
iter_=0
for k,t in enumerate(delays):
    for i in range(len(label)):
        for j in range(3):
            tmp = np.roll(ref_traj[i,j,:],t)
            tmp[:t]=0
            ref_traj_rolled[iter_,j,:] = tmp
        ref_label_rolled[iter_] = label[i]
        iter_ = iter_+1

flipped = np.copy(ref_traj_rolled)
flipped[:,[1,2]] = -flipped[:,[1,2]] # We only flip y and bodyangle to get the symetric bouts
ref_bouts = np.vstack((ref_traj_rolled,flipped))

ref_labels = np.concatenate((ref_label_rolled,ref_label_rolled+13))

NameCat = [i+'_+' for i in NameCat]+[i+'_-' for i in NameCat]
print(NameCat)
ref_bouts_flat = np.reshape(ref_bouts,(ref_bouts.shape[0],ref_bouts.shape[1]*ref_bouts.shape[2]))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(ref_bouts_flat, ref_labels)


###################################################################
#################     Load All Bouts     ##########################
###################################################################

# Mapping between ROI and Drug:
Drugs = ['Ketanserin','Quipazin','Trazodone','Quinpirole','Clozapine','Ketanserin',
'Quipazin','Trazodone','Quinpirole','Clozapine','Fluoxetine','Haloperidol',
'Apomorphine','Valpoic','Ctrl','Fluoxetine','Haloperidol','Apomorphine','Valpoic','Ctrl-DMSO']

def load_bout_cat_from_fish(FishTable,id_fish,ref_bouts_flat,ref_labels):
    fish = Fish(FishTable.iloc[id_fish])
    camlog,stimlog = fish.load_all_matlog()

    Traj = TrajectoryPreprocessing(fish,position_column=['fish_blob_c','fish_blob_l'],angle_column='body_angle_cart',
                                    out_xy_style='cartesian',
                                    out_xy_unit='mm',
                                    out_time_unit='camera_frame')
    tracking_df,circle = Traj.get_trajectory_df(camlog)
    tracking_df = add_hierarchical_level(tracking_df,'trajectory')

    discrete_visual_stim = {'UniformLight': ['color'],
                            'RotationOMR': ['grating_direction','grating_speed'],
                            'ApproachingDot':['dot_direction']}

    continuous_visual_stim = {'UniformLight': [],
                            'ApproachingDot': [],
                            'RotationOMR':[]}
                            
    shader_list = [s for s in discrete_visual_stim.keys()]

    default_uniforms=['frame_id','iGlobalTime','f'+f'{fish.roi:02d}'+'x','f'+f'{fish.roi:02d}'+'y','f'+f'{fish.roi:02d}'+'t']

    stim = ShaderPreprocessing(shader_list,discrete_visual_stim,continuous_visual_stim,default_uniforms)
    visual_stim_df = stim.get_visual_stim_df(stimlog)
    tracking_df = add_hierarchical_level(tracking_df,'tracking')
    ethogram_df = pd.merge(tracking_df,visual_stim_df,left_on=[('frame_id')], how='left',right_on=[('visual_stimuli','synch', 'frame_id_interp')])

    stim = EthogramStimuliProcessing()

    #################################
    ######### COMPUTE SCALARS #######
    #################################

    ethogram_df_flat=ethogram_df.copy()
    # Flatten
    ethogram_df_flat.columns = ['_'.join(col) for col in ethogram_df.columns]
    fps_index = fish.fps

    radius = 22
    fps = 400


    x = ethogram_df_flat['tracking_trajectory_x'].iloc[:].values
    y = -ethogram_df_flat['tracking_trajectory_y'].iloc[:].values
    body_angle = ethogram_df_flat['tracking_trajectory_angle'].iloc[:].values+np.pi

    #bout_cat,onset_aligned,offset_aligned = find_and_classify_bouts(x,y,body_angle)

    ## Compute Lateral and Axial speed:
    body_vector = np.array([np.cos(body_angle),np.sin(body_angle)])[:,:-1]
    x = one_euro_filter(x,10,2,fps)
    y = one_euro_filter(y,10,2,fps)
    body_angle = one_euro_filter(body_angle,10,2,fps)

    distance_to_center = np.sqrt(x*x+y*y)
    '''
    angular_speed = diff_but_better(body_angle,dt=1/fps,filter_length=75)
    speed=np.sqrt(diff_but_better(x,dt=1/fps,filter_length=75)**2+diff_but_better(y,dt=1/fps,filter_length=75)**2)

    speed[np.isnan(speed)] = 0
    angular_speed[np.isnan(angular_speed)] = 0
    speed = np.array([0]+speed.tolist())
    angular_speed = np.array([0]+angular_speed.tolist())

    ethogram_df_flat['speed'] = speed
    ethogram_df_flat['angular_speed'] = angular_speed
    ethogram_df_flat['distance_to_center'] = distance_to_center
    '''
    #################################
    ######### COMPUTE STIMULI #######
    #################################
    # Dictionnary: Trial => Stim => Condition (column) => Value => Onset/Offset
    trials={}

    trials['ApproachingDot'] = {'column':'visual_stimuli_ApproachingDot_dot_direction','value':[-90,90],'interval_margin':[-3,3]}

    trials['RotationOMR'] = {'column':'visual_stimuli_RotationOMR_grating_direction','value':[0, 45, 90, 135, 180, 225, 270, 315],'interval_margin':[-5,0]}

    trials['LightDark'] = {'column':'visual_stimuli_UniformLight_color','value':[0],'interval_margin':[0,0]}

    trials['Spontaneous'] = {'column':'visual_stimuli_UniformLight_color','value':[1],'interval_margin':[0,0]}
    #
    #tracking = EthogramTrackingProcessing()

    # Extract Stimuli:
    stimuli_df = ethogram_df_flat['visual_stimuli_synch_shader_name'].copy()
    stimuli_df.iloc[:] = str('no_salient_stim')
    stimuli_df.astype('category',copy=True)

    for trial_key in trials.keys():#= 'RotationOMR'
        condition = trials[trial_key]['column']
        shader = condition.split('_')[2]
        uniform_condition='_'.join(condition.split("_")[3:])

        ##################
        # Loop on value  #
        ##################
        #print(condition)

        for i_val,val in enumerate(trials[trial_key]['value']):

            #################
            # Find interval #
            #################
            onset,offset,duration = stim.find_uniform_value(ethogram_df_flat,
                                                                shader=shader,
                                                                uniform_condition=uniform_condition,
                                                                uniform_value=val,
                                                                min_duration=50,
                                                                max_duration=15*60*fps_index)

            margin_m = fps_index*trials[trial_key]['interval_margin'][0]
            margin_p = fps_index*trials[trial_key]['interval_margin'][1]
            onset_w_margin = [ o - margin_m for o in onset ]
            offset_w_margin = [ o + margin_p for o in offset ]
            if trial_key == 'LightDark':
                offset_w_margin = [ o + fps_index*10 for o in onset_w_margin ]

            # Change offset if we only care about short duration:
            duration=[(off-on)/fps_index for off,on in zip(offset_w_margin,onset_w_margin)]

            if trial_key == 'Spontaneous':
                    onset_w_margin = [ on for on,dur in zip(onset_w_margin,duration) if dur>13*60]
                    offset_w_margin = [ off for off,dur in zip(offset_w_margin,duration) if dur>13*60]
                    duration=[(off-on)/fps_index for on,off in zip(onset_w_margin,offset_w_margin)]

            for on,off in zip(onset_w_margin,offset_w_margin):
                stimuli_df.loc[on:off] = str(trial_key)+str(val)

    stimuli_df = stimuli_df.astype('category',copy=True)


    #################################
    ####### CROP TO FINISH EXP ######
    #################################
    onset,offset,duration = find_onset_offset_numpy(stimuli_df.values=='Spontaneous1')
    ethogram_df_flat = ethogram_df_flat.iloc[:offset[2]]
    stimuli_df = stimuli_df.iloc[:offset[2]]

    #################################
    ######### COMPUTE BoutCat #######
    #################################

    x = ethogram_df_flat['tracking_trajectory_x'].iloc[:].values
    y = -ethogram_df_flat['tracking_trajectory_y'].iloc[:].values
    body_angle = ethogram_df_flat['tracking_trajectory_angle'].iloc[:].values+np.pi

    #bout_cat,onset_aligned,offset_aligned = find_and_classify_bouts(x,y,body_angle)

    ## Compute Lateral and Axial speed:
    body_vector = np.array([np.cos(body_angle),np.sin(body_angle)])[:,:-1]
    x = one_euro_filter(x,10,2,fps)
    y = one_euro_filter(y,10,2,fps)
    body_angle = one_euro_filter(body_angle,10,2,fps)

    from utils import diff_but_better
    Ndiff=45
    position_change = np.zeros_like(body_vector)
    position_change[0,:] = diff_but_better(x,dt=1/fps, filter_length=Ndiff)[:-1]
    position_change[1,:] = diff_but_better(y,dt=1/fps, filter_length=Ndiff)[:-1]
    angle = np.pi/2
    rotMat = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    body_vector_orth = np.dot(rotMat,body_vector)

    axial_speed = np.einsum('ij,ij->j',body_vector,position_change)
    lateral_speed = np.einsum('ij,ij->j',body_vector_orth,position_change)
    yaw_speed = diff_but_better(np.unwrap(body_angle),dt=1/fps, filter_length=Ndiff)[:-1]
    axial_speed=np.concatenate((np.array([np.nan]),axial_speed))
    lateral_speed=np.concatenate((np.array([np.nan]),lateral_speed))
    yaw_speed=np.concatenate((np.array([np.nan]),yaw_speed))
    speed_amplitude = np.sqrt(np.power(axial_speed,2)+np.power(lateral_speed,2))

    low_pass_tail_speed,max_filt,min_filt = mexican_hat_tail_speed(np.abs(yaw_speed),MinFiltSize=200,MaxFiltSize=10)
    #BoutThresh = estimate_speed_threshold(low_pass_tail_speed,margin_std=15, bin_log_min = -5, bin_log_max=5 )
    BoutThresh = 2.
    Max_ISI=20
    kernel = np.ones(Max_ISI)

    filtered_timeforward = np.convolve(kernel,low_pass_tail_speed>BoutThresh, mode='full')[:low_pass_tail_speed.shape[0]] # Trick to make convolution causal
    bout_indicator = 1.0*((filtered_timeforward)>0)#*filtered_timebackward)>0)

    onset,offset,duration = find_onset_offset_numpy(bout_indicator)

    onset = np.array(onset)
    offset = np.array(offset)
    onset,offset = onset[(onset+60)<len(x)],offset[(onset+60)<len(x)]

    onset_aligned = []
    offset_aligned = []
    all_bouts = np.zeros((0,60))
    for on_,off_,dur_ in zip(onset,offset,duration):
        data = yaw_speed[on_:off_]
        try:
            peak_loc = align_bout_peaks(data)
            if np.isnan(peak_loc)==False:
                on_new = on_+peak_loc-10
                if on_new>0:
                    onset_aligned.append(on_new)
                    offset_aligned.append(off_)
                    tmp = body_angle[on_new:on_new+60]
                    tmp = tmp -tmp[0]
                    all_bouts = np.vstack((all_bouts,tmp))
            else:
                onset_aligned.append(on_-20)
                offset_aligned.append(off_)
                tmp = body_angle[on_:on_+60]
                tmp = tmp -tmp[0]
                all_bouts = np.vstack((all_bouts,tmp))
        except:
            onset_aligned.append(on_-20)
            offset_aligned.append(off_)
            tmp = body_angle[on_:on_+60]
            tmp = tmp -tmp[0]
            all_bouts = np.vstack((all_bouts,tmp))

            

    onset_aligned = np.array(onset_aligned)
    offset_aligned = np.array(offset_aligned)
    onset_aligned,offset_aligned = onset_aligned[(onset_aligned+60)<len(x)],offset_aligned[(onset_aligned+60)<len(x)]

    #bout_cat = find_nearest_bouts_parallel(onset_aligned,x,y,body_angle,ref_bouts_flat,ref_labels)
    #bout_cat = np.array(bout_cat)


    bout_array = np.zeros((len(onset_aligned),80*3))
    for i,(on_,off_) in enumerate(zip(onset_aligned,offset_aligned)):
        sub_x,sub_y,sub_body_angle = x[on_-20:on_+60],y[on_-20:on_+60],body_angle[on_-20:on_+60]
        Pos = np.zeros((2,80))
        Pos[0,:] = sub_x-sub_x[0]
        Pos[1,:] = sub_y-sub_y[0]
        theta=-sub_body_angle[0]
        body_angle_rotated=sub_body_angle-sub_body_angle[0]
        RotMat=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        PosRot=np.dot(RotMat,Pos)
        sub_x,sub_y,sub_body_angle = PosRot[0,:],PosRot[1,:],body_angle_rotated
        bout_array[i,:80] = sub_x
        bout_array[i,80:80*2] = sub_y
        bout_array[i,80*2:] = sub_body_angle


    bout_cat = knn.predict(bout_array)
    '''
    #D = distance.cdist(bout_array, ref_bouts_flat, 'euclidean')
    
    X1,X2 = torch.from_numpy(bout_array),torch.from_numpy(ref_bouts_flat)
    X1,X2 = X1.to(device),X2.to(device)

    D = torch.cdist(X1,X2, p=2)
    D = D.clone().detach().cpu().numpy()

    bout_cat = ref_labels[np.argmin(D,axis=1)]'''

    bout_cat_ts = np.zeros_like(body_angle)-1
    for i,(on_,off_) in enumerate(zip(onset_aligned,offset_aligned)):
        bout_cat_ts[on_:off_]=bout_cat[i]

    #################################
    ######## COLLECT HISTOGRAM ######
    #################################

    #all_stim = np.setdiff1d(stimuli_df.values.unique(),'no_salient_stim')
    all_stim = ['Spontaneous1','ApproachingDot-90', 'ApproachingDot90', 'LightDark0', 'RotationOMR0',
    'RotationOMR135', 'RotationOMR180', 'RotationOMR225', 'RotationOMR270',
    'RotationOMR315', 'RotationOMR45', 'RotationOMR90']

    Freq_All_Stim = np.zeros((len(all_stim),26+1))
    for i,s in enumerate(all_stim):
        onset,offset,duration = find_onset_offset_numpy(stimuli_df.values==s)

        freq = np.zeros(26+1)
        for on_,off_ in zip(onset,offset):
            sub_boutcat_ts = bout_cat_ts[on_:off_]
            count = np.histogram(sub_boutcat_ts,np.arange(-1,27)-0.5)[0]
            #id_b = np.where((onset_aligned>on_)&(onset_aligned<off_))[0]
            #count = np.histogram(bout_cat[id_b],np.arange(27)-0.5)[0]
            T = (off_-on_)/400
            freq = freq + count/(T)
        freq = freq/len(onset)
        Freq_All_Stim[i,:] = freq
    
    return bout_cat,onset_aligned,offset_aligned,Freq_All_Stim,stimuli_df


dict_ = {}
k = 0
for i in range(0,FishTable.shape[0]):
    dr = Drugs[FishTable.iloc[i].roi]
    concentr_ = FishTable.iloc[i].drug_concentration
    expe_name = FishTable.iloc[i].expe_name
    if concentr_=='HIGH' or concentr_=='MID':
        try:
            print(i)
            print(FishTable.iloc[i].expe_name)
            bout_cat,onset_aligned,offset_aligned,Freq_All_Stim,stimuli_df = load_bout_cat_from_fish(FishTable,i,ref_bouts_flat,ref_labels)
            dict_[k]={'Treatment':dr,'Concentration':concentr_,'expe_name':expe_name,'Freq_All_Stim':Freq_All_Stim,'bout_cat':bout_cat,'onset_aligned':onset_aligned,'offset_aligned':offset_aligned,'stimuli_df':stimuli_df}
            k = k+1
        except:
            print(len(dict_))
            pass


import pickle
# Use pickle to save the dict
filename_results='Super_resolved_Dataset_wkNN.pickle'
with open(filename_results, 'wb') as handle:
    pickle.dump(dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)
