import numpy as np
#from scipy.ndimage import zoom
from super_resolution.downsampling import create_downsampling_function
from scipy.ndimage.interpolation import shift
from dataclasses import dataclass,field
from utils.utils_bouts import NameCat

@dataclass
class Template:
    """This is a dataclasse for template bouts.

    Attributes:
        BoutDuration (int): An integer.
        bouts (np.ndarray): template bouts of shape (n_samples, n_features, BoutDuration)
        delays (np.ndarray): shift corresponding to each bouts of shape (n_samples)
        label (np.ndarray): category integer corresponding to each bouts of shape (n_samples)
        label_str (np.ndarray): category name corresponding to each bouts of shape (n_samples)
        
    """    
    BoutDuration: int
    bouts: np.ndarray=field(repr=False)
    delays: np.ndarray=field(repr=False)
    label: np.ndarray=field(repr=False)
    label_str: np.ndarray=field(repr=False)
    
    def flat(self) -> np.ndarray:
        return np.reshape(self.bouts,(self.bouts.shape[0],self.bouts.shape[1]*self.bouts.shape[2]))

#TODO: Need to change template.npz such that the templates are already balanced and normalized
#TODO: Remove hardcoded NameCat
#TODO: Only use full format

def generate_template_bouts(format='tail&traj',target_fps=700,ExludeCaptureSwim=True,delays=np.arange(0,60,9)):
    """Generate reference movements for classifying new movments

    Args:
        format (str, optional): _description_. Defaults to 'tail&traj'.
        target_fps (int, optional): _description_. Defaults to 700.
        ExludeCaptureSwim (bool, optional): _description_. Defaults to True.
        delays (_type_, optional): _description_. Defaults to np.arange(0,60,9).

    Returns:
        _type_: Template dataclass
    """    
    arr = np.load('./classification/CleanBalancedBoutDataset.npz')
    ref_bouts = arr['bouts']
    labels = arr['label']

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
                templates_rolled[iter_,j,:] = shift(templates[i,j,:],t,mode='nearest')# cval=0)
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
    
