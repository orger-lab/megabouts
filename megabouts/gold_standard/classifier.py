import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsClassifier


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

