from dataclasses import dataclass,field
import numpy as np
from skimage.transform import resize

import warnings
  
@dataclass
class Dataset_FullTracking():
    fps: int = field(init=True,repr=True)
    x: np.ndarray=field(init=True,repr=False)
    y: np.ndarray=field(init=True,repr=False)
    body_angle: np.ndarray=field(init=True,repr=False)
    tail_angle: np.ndarray=field(init=True,repr=False)
    tracking_type: str = field(repr=True,default='tail_and_traj')

    def __post_init__(self):

        if self.tail_angle.shape[1]>self.tail_angle.shape[0]:
            self.tail_angle = self.tail_angle.T
            print(f"tail_angle was flipped to have size {self.tail_angle.shape[0]},{self.tail_angle.shape[1]}")

        same_lenght = (len(self.x) == len(self.y) == len(self.body_angle) == self.tail_angle.shape[0])
        if not same_lenght:
            raise ValueError("All input must have the same number of time steps")

        if self.tail_angle.shape[0]<(60*self.fps):
            warnings.warn("The pipeline should be applied to long recording to allow for automatic segmentation and avoid border effect")
    @property
    def n_frames(self):
        return len(self.x)


@dataclass
class Dataset_TailTracking():
    fps: int = field(init=True,repr=True)
    tail_angle: np.ndarray=field(init=True,repr=False)
    tracking_type: str = field(repr=True,default='tail_and_traj')

    def __post_init__(self):

        if self.tail_angle.shape[1]>self.tail_angle.shape[0]:
            self.tail_angle = self.tail_angle.T
            print(f"tail_angle was flipped to have size {self.tail_angle.shape[0]},{self.tail_angle.shape[1]}")

        if self.tail_angle.shape[0]<(60*self.fps):
            warnings.warn("The pipeline should be applied to long recording to allow for automatic segmentation and avoid border effect")

    @property
    def n_frames(self):
        return self.tail_angle.shape[0]


@dataclass
class Dataset_CentroidTracking():
    fps: int = field(init=True,repr=True)
    x: np.ndarray=field(init=True,repr=False)
    y: np.ndarray=field(init=True,repr=False)
    body_angle: np.ndarray=field(init=True,repr=False)
    tracking_type: str = field(repr=True,default='tail_and_traj')

    def __post_init__(self):

        same_lenght = (len(self.x) == len(self.y) == len(self.body_angle))
        if not same_lenght:
            raise ValueError("All input must have the same number of time steps")

        if self.tail_angle.shape[0]<(60*self.fps):
            warnings.warn("The pipeline should be applied to long recording to allow for automatic segmentation and avoid border effect")

    @property
    def n_frames(self):
        return len(self.x)