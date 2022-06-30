import numpy as np
#from scipy.ndimage import zoom
from super_resolution.downsampling import create_downsampling_function
from scipy.ndimage.interpolation import shift
from dataclasses import dataclass,field
from utils.utils_bouts import NameCat
 

@dataclass(repr=False)
class Knn_Training_Dataset():

    fps: int
    augmentation_delays: np.ndarray
    ignore_CS: bool = True
    bout_duration: int = field(init=False)
    _bouts: np.ndarray = field(init=False)
    labels: np.ndarray = field(init=False)
    labels_str : np.ndarray = field(init=False)
    delays : np.ndarray = field(init=False)

    def __post_init__(self):
        
        arr = np.load('./classification/kNN_Training_Dataset.npz')
        bouts = arr['bouts']
        labels = arr['labels']
        if self.ignore_CS:
            bouts = bouts[(labels!=3)&(labels!=4)] # Remove slow and fast capture swim
            labels = labels[(labels!=3)&(labels!=4)]  # Remove slow and fast capture swim

        bouts = self._downsample(bouts)
        bouts,labels,delays = self._augment_with_delays(bouts,labels)
        bouts,labels,delays = self._augment_with_negative(bouts, labels,delays)

        self._bouts=bouts
        self.labels = labels.astype('int')
        self.labels_str = 0#np.array(NameCat)[labels]
        self.delays = delays
        self.bout_duration = self._bouts.shape[2]

    @property
    def tail_and_traj(self):
        return self._bouts

    @property    
    def tail(self):
        return self._bouts[:,:7,:]
    
    @property
    def traj(self):
        return self._bouts[:,7:,:]

    @property
    def tail_and_traj_flat(self):
        return self._flatten(self.tail_and_traj)
    
    @property
    def tail_flat(self):
        return self._flatten(self.tail)
    
    @property
    def traj_flat(self):
        return self._flatten(self.traj)

    def _downsample(self,bouts,axis=2):
        downsampling_f, Duration_after_Downsampling,t,tnew = create_downsampling_function(fps_new=self.fps,n_frames_origin=140,fps_origin=700)
        return downsampling_f(bouts,axis=axis)

    def _augment_with_delays(self,bouts,labels):
        N_delay = len(self.augmentation_delays)
        N_bouts = bouts.shape[0]
        N_feat = bouts.shape[1]
        N_timestep = bouts.shape[2]

        bouts_rolled = np.zeros((N_bouts*N_delay,N_feat,N_timestep))
        labels_rolled,delays_rolled = np.nan*np.ones(N_bouts*N_delay),np.nan*np.ones(N_bouts*N_delay)
        
        iter_=0
        for t in self.augmentation_delays:
            bouts_rolled[iter_:iter_+N_bouts,:,:] = shift(bouts,(0,0,t),mode='nearest')
            labels_rolled[iter_:iter_+N_bouts] = labels
            delays_rolled[iter_:iter_+N_bouts] = t
            iter_ = iter_+N_bouts
        
        return bouts_rolled,labels_rolled,delays_rolled
    
    def _augment_with_negative(self,bouts,labels,delays):
        flipped = np.copy(bouts)
        flipped[:,[0,1,2,3,4,5,6,8,9]] = -flipped[:,[0,1,2,3,4,5,6,8,9]] # We only tail, flip y and bodyangle to get the symetric bouts
        
        bouts = np.vstack((bouts,flipped))
        labels = np.concatenate((labels,labels+13))
        delays = np.concatenate((delays,delays))
        return bouts,labels,delays

    def _flatten(self,array):
        array_flat = np.reshape(array,(array.shape[0],array.shape[1]*array.shape[2]))
        return array_flat
