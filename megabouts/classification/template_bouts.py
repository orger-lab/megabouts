import numpy as np
#from scipy.ndimage import zoom
from megabouts.utils.utils_downsampling import create_downsampling_function,convert_ms_to_frames
from scipy.ndimage.interpolation import shift
from dataclasses import dataclass,field
from pathlib import Path


@dataclass(repr=False)
class Knn_Training_Dataset():
    

    fps: int
    augmentation_delays: np.ndarray
    bout_duration: int = field(init=True)
    peak_loc: int = field(init=True)
    
    bouts_dict: dict = field(init=True)
    _bouts: np.ndarray = field(init=False)
    labels: np.ndarray = field(init=False)
    labels_str : np.ndarray = field(init=False)
    delays : np.ndarray = field(init=False)
    
    def __post_init__(self):
        
        bouts = np.concatenate((self.bouts_dict['tail'],self.bouts_dict['traj']),axis=1)
        labels = self.bouts_dict['labels']
        
        bouts = self._downsample(bouts)
        print(bouts.shape)
        bouts,labels,delays = self._augment_with_delays(bouts,labels)
        bouts,labels,delays = self._augment_with_negative(bouts, labels,delays)

        self._bouts = bouts
        self.labels = labels.astype('int')
        self.delays = delays

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
        fps_origin = self.bouts_dict['params']['fps']
        #bouts_duration_origin_ms = self.bouts_dict['params']['bout_duration']*1000/fps_origin
        #downsampling_f, Duration_after_Downsampling,t,tnew = create_downsampling_function(fps_new=self.fps,fps_origin=fps_origin,duration_ms=bouts_duration_origin_ms)
        downsampling_f, Duration_after_Downsampling,t,tnew = create_downsampling_function(fps_new=self.fps,fps_origin=fps_origin,duration=self.bouts_dict['params']['bout_duration'],duration_unit='frames',kind='linear')
        
        return downsampling_f(bouts,axis=axis)

    def _augment_with_delays(self,bouts,labels):
        
        original_bout_duration = self.bouts_dict['params']['bout_duration']
        original_bout_duration_ms = self.bouts_dict['params']['bout_duration']
        target_bout_duration = self.bout_duration
        
        original_peak_loc_ms = self.bouts_dict['params']['peak_loc']*1000/self.bouts_dict['params']['fps']
        original_peak_loc = convert_ms_to_frames(self.fps,original_peak_loc_ms) 
        target_peak_loc = self.peak_loc
        
        MaxPosShift = original_bout_duration-target_bout_duration
        MinPeakLoc = max(1,original_peak_loc-MaxPosShift) # We don't allow to miss the first peak so it as to be located at 1 minimum
        MaxPeakLoc = original_peak_loc
        # augmentation_FirstHalfBeatLoc = np.arange(MinPeakLoc,MaxPeakLoc)
        # augmentation_delays =augmentation_FirstHalfBeatLoc-20
        
        # Prepare Variable to store delayed tail
        N_delay = len(self.augmentation_delays)
        N_bouts = bouts.shape[0]
        N_feat = bouts.shape[1]
        N_timestep = self.bout_duration
        
        bouts_rolled = np.zeros((N_bouts*N_delay,N_feat,N_timestep))
        labels_rolled,delays_rolled = np.nan*np.ones(N_bouts*N_delay),np.nan*np.ones(N_bouts*N_delay)
        
        iter_=0
        print(f'TimeStep:{N_timestep},OriginalPeakLoc:{original_peak_loc},TargetPeak:{target_peak_loc}')
        print(f'augmentation_delays:{self.augmentation_delays}')

        for t in self.augmentation_delays:
            id_st = (original_peak_loc-t)-target_peak_loc
            print(f'IdSt:{id_st}')
            bouts_rolled[iter_:iter_+N_bouts,:,:] = shift(bouts,(0,0,t),mode='nearest')[:,:,original_peak_loc-target_peak_loc:original_peak_loc-target_peak_loc+N_timestep]

            #if (id_st+N_timestep)<bouts.shape[2]:
            #    bouts_rolled[iter_:iter_+N_bouts,:,:] = bouts[:,:,id_st:id_st+N_timestep] 
            #else:
            #    bouts_rolled[iter_:iter_+N_bouts,:,:] = shift(bouts[:,:,original_peak_loc-target_peak_loc:original_peak_loc-target_peak_loc+N_timestep],(0,0,t),mode='nearest')
            
            labels_rolled[iter_:iter_+N_bouts] = labels
            delays_rolled[iter_:iter_+N_bouts] = t
            iter_ = iter_+N_bouts        
        
        traj_aligned = np.copy(bouts_rolled[:,7:,:])

        for i in range(traj_aligned.shape[0]):
                
                sub_x,sub_y,sub_body_angle = traj_aligned[i,0,:],traj_aligned[i,1,:],traj_aligned[i,2,:]
                Pos = np.zeros((2,len(sub_x)))
                Pos[0,:] = sub_x-sub_x[0]
                Pos[1,:] = sub_y-sub_y[0]
                theta=-sub_body_angle[0]
                body_angle_rotated=sub_body_angle-sub_body_angle[0]
                RotMat=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                PosRot=np.dot(RotMat,Pos)
                sub_x,sub_y,sub_body_angle = PosRot[0,:],PosRot[1,:],body_angle_rotated

                traj_aligned[i,0,:],traj_aligned[i,1,:],traj_aligned[i,2,:] = sub_x,sub_y,sub_body_angle
            
        bouts_rolled[:,7:,:] = traj_aligned
                
        return bouts_rolled,labels_rolled,delays_rolled
    
    def _augment_with_negative(self,bouts,labels,delays):
        num_cat = len(np.unique(labels))
        flipped = np.copy(bouts)
        flipped[:,[0,1,2,3,4,5,6,8,9]] = -flipped[:,[0,1,2,3,4,5,6,8,9]] # We only tail, flip y and bodyangle to get the symetric bouts
        
        bouts = np.vstack((bouts,flipped))
        labels = np.concatenate((labels,labels+num_cat))
        delays = np.concatenate((delays,delays))
        return bouts,labels,delays

    def _flatten(self,array):
        array_flat = np.reshape(array,(array.shape[0],array.shape[1]*array.shape[2]))
        return array_flat
