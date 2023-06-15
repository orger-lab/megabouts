
import cv2
import numpy as np
from skimage import measure
from skimage import filters
from skimage.measure import label, regionprops,regionprops_table
from scipy import signal


def find_half_beat(bout_slice,half_BC_filt = 150, std_thresh = 5,min_size_blob = 500):

    BC_filt = half_BC_filt*2+1
    X = bout_slice
    oversample_slice = cv2.resize(X, dsize=(X.shape[1],X.shape[0]*10), interpolation=cv2.INTER_CUBIC)
        
    filtered_=np.zeros_like(oversample_slice)
    binary_thresh_up=np.zeros_like(oversample_slice)
    binary_thresh_down=np.zeros_like(oversample_slice)
    for i in range(oversample_slice.shape[1]):
            filtered_[:,i]= signal.convolve(oversample_slice[:,i],1/BC_filt*signal.boxcar(M=BC_filt,sym=True),mode='same')
            sigma = np.std(filtered_[:,i])
            binary_thresh_up[:,i] = filtered_[:,i]+sigma/std_thresh
            binary_thresh_down[:,i] = filtered_[:,i]-sigma/std_thresh

    binary_image = np.zeros_like(oversample_slice)
    for i in range(oversample_slice.shape[1]):
        binary_image[:,i] = (oversample_slice[:,i]>binary_thresh_up[:,i]) + -1*(oversample_slice[:,i]<binary_thresh_down[:,i])

    all_labels_pos = measure.label((binary_image)==1)
    if len(np.where(all_labels_pos)[0])>0:
        props_pos = regionprops_table(all_labels_pos, properties=('area','centroid'))
        half_beat_pos = []
        for i,lab_ in enumerate(np.unique(all_labels_pos)[1:]):
            if props_pos['area'][i]> min_size_blob:
                id = np.where(all_labels_pos[:,-1]==lab_)[0]
                if len(id)>0:
                    half_beat_pos.append(id[np.argmax(oversample_slice[id,-1])])
    else:
        half_beat_pos = []

    all_labels_neg = measure.label((binary_image)==-1)
    if len(np.where(all_labels_neg)[0])>0:
        props_neg = regionprops_table(all_labels_neg, properties=('area','centroid'))
        half_beat_neg = []
        for i,lab_ in enumerate(np.unique(all_labels_neg)[1:]):
            if props_neg['area'][i]> min_size_blob:
                id = np.where(all_labels_neg[:,-1]==lab_)[0]
                if len(id)>0:
                    half_beat_neg.append(id[np.argmin(oversample_slice[id,-1])])
    else:
        half_beat_neg = []


    half_beat_pos = np.floor(np.array(half_beat_pos)/10).astype('int')
    half_beat_neg = np.floor(np.array(half_beat_neg)/10).astype('int')

    
    return half_beat_pos,half_beat_neg,binary_image
