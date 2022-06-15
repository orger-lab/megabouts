import numpy as np
#from scipy.ndimage import zoom
from super_resolution.downsampling import create_downsampling_function
from scipy.ndimage.interpolation import shift

def generate_template_bouts(format='tail&traj',target_fps=700,ExludeCaptureSwim=True,delays=np.arange(0,60,9)):
    
    arr = np.load('./classification/CleanBalancedBoutDataset.npz')
    ref_bouts = arr['bouts']
    labels = arr['label']
    NameCat = ['approach_swim','slow1','slow2','slow_capture_swim','fast_capture_swim','burst_swim','J_turn','high_angle_turn','routine_turn','spot_avoidance_turn','O_bend','long_latency_C_start','C_start']

    N_Sample = 1400
    balanced_bouts = np.zeros((0,ref_bouts.shape[1]))
    balanced_labels = []
    for b in range(13):
        id_ = np.where(labels==b)[0]
        np.random.seed(42)
        id_balanced = np.random.permutation(id_)[:N_Sample]
        balanced_bouts = np.vstack((balanced_bouts,ref_bouts[id_balanced,:]))
        balanced_labels = balanced_labels + [b]*N_Sample

    balanced_labels = np.array(balanced_labels)

    ref_bouts = balanced_bouts 
    labels = balanced_labels

    ref_bouts = ref_bouts[(labels!=3)&(labels!=4)] # Remove slow and fast capture swim
    labels = labels[(labels!=3)&(labels!=4)]  # Remove slow and fast capture swim

    downsampling_f, Duration_after_Downsampling,t,tnew = create_downsampling_function(target_fps,duration_original=140,original_fps=700)

    templates = np.zeros((len(labels),10,Duration_after_Downsampling))
    if format=='tail':
        templates = np.zeros((len(labels),7,Duration_after_Downsampling))
    if format=='traj':
        templates = np.zeros((len(labels),3,Duration_after_Downsampling))

    for i in range(len(labels)):
        tmp = ref_bouts[i,:]
        tmp = np.reshape(tmp,(10,140))
        if format=='tail':
            tmp = tmp[:7,:]  # Only use the trajectory
        if format=='traj':
            tmp = tmp[7:,:]  # Only use the trajectory

        #for j in range(templates.shape[1]):
        #    templates[i,j,:]= zoom(tmp[j,:],downsampling_factor,order=3)
        templates[i,:,:]= downsampling_f(tmp[:,:],axis=1)
        
    if format=='traj':
        templates[:,0,:],templates[:,1,:],templates[:,2,:] = templates[:,0,:]*5,templates[:,1,:]*5,templates[:,2,:]/2 # Rescaling to make x,y in mm and angle in radian
    if format=='tail&traj':
        templates[:,7,:],templates[:,8,:],templates[:,9,:] = templates[:,7,:]*5,templates[:,8,:]*5,templates[:,9,:]/2 # Rescaling to make x,y in mm and angle in radian

    templates_rolled = np.zeros((len(labels)*len(delays),templates.shape[1],Duration_after_Downsampling))
    labels_rolled = np.nan*np.ones(len(labels)*len(delays))
    delays_rolled = np.nan*np.ones(len(labels)*len(delays))

    iter_=0
    for k,t in enumerate(delays):
        for i in range(len(labels)):
            for j in range(templates.shape[1]):
                #tmp = np.roll(templates[i,j,:],t)
                #tmp[:t]=0
                templates_rolled[iter_,j,:] = shift(templates[i,j,:],t, cval=0)
            labels_rolled[iter_] = labels[i]
            delays_rolled[iter_] = t
            iter_ = iter_+1

    flipped = np.copy(templates_rolled)
    if format == 'traj':
        flipped[:,[1,2]] = -flipped[:,[1,2]] # We only flip y and bodyangle to get the symetric bouts
    if format == 'tail':
        flipped = -flipped
    if format == 'tail&traj':
        flipped[:,[0,1,2,3,4,5,6,8,9]] = -flipped[:,[0,1,2,3,4,5,6,8,9]] # We only tail, flip y and bodyangle to get the symetric bouts


    templates = np.vstack((templates_rolled,flipped))

    labels = np.concatenate((labels_rolled,labels_rolled+13))
    delays = np.concatenate((delays_rolled,delays_rolled))

    NameCat = [i+'_+' for i in NameCat]+[i+'_-' for i in NameCat]
    templates_flat = np.reshape(templates,(templates.shape[0],templates.shape[1]*templates.shape[2]))

    return templates_flat,labels,delays,NameCat,Duration_after_Downsampling
    
    '''


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(ref_bouts_flat, ref_labels)


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
