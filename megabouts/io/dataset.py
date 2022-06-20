from dataclasses import dataclass,field
import numpy as np



'''
Bouts(fps,init_onset,init_offset):

CentroidBouts
Attribute: Centroid + CentroidCentered

TailBouts
Attribute: TailAngle

Method: Get Segment Matrix

'''


# USING CACHE PROPERTY FOR PIPELINE


class DataSet:

    def __init__(self, sequence_of_numbers):
        self._data = tuple(sequence_of_numbers)

    @cached_property
    def stdev(self):
        return statistics.stdev(self._data)


'''
TrackingBase (fps)

TrackingCentroid()
Attribute: Position+HeadAngle

TrackingTail()
Attribute: TailAngle

TrackingFull(TrackingCentroid,TrackingTail)
Attribute: TailAngle + Centroid

Method: Smooth
Method: Segment(self,onset,offset) (specific to tail/centroid or full)
'''


@dataclass
class TrackingBase:
    """Base dataclasse for tracking data.

    Attributes:
        fps (int): fps of recording
        n_frames (int): number of frames of recording
        kind (str): can be 'full' 'tail' or 'centroid'
    """    
    fps: int
    n_frames: int = field(init=False)

@dataclass
class TailTracking(TrackingBase):
    """Base dataclasse for tracking data.

    Attributes:
        tail_angle (np.ndarray): tail angle of size (n_frames,)
    """    
    tail_angle: np.ndarray=field(repr=False)
    

@dataclass
class CentroidTracking(TrackingBase):
    position: np.ndarray=field(repr=False)
    body_angle: np.ndarray=field(repr=False)
    