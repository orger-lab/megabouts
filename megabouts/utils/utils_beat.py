import numpy as np
import pandas as pd

#from noise_robust_differentiator import derivative_n2
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


def is_there_signal(x,Thresh=2):

    x = (x-np.mean(x))/np.std(x)
    y_smooth = savgol_filter(x,7, 2)
    NoiseLevel = 100*np.std(x-y_smooth)
    return NoiseLevel<Thresh

def refine_segmentation_bouts(onset,offset,Min_Duration = 40, Min_IBI = 10):
    
    duration = offset - onset
    
    onset = onset[duration>Min_Duration]
    offset = offset[duration>Min_Duration]
    duration = duration[duration>Min_Duration]
    
    inter_bouts = [on_-off_ for (on_,off_) in zip(onset[1:],offset[:-1])]
    # Merge Bouts with too small inter_bouts_interval:
    id = np.where(np.array(inter_bouts)<Min_IBI)[0]
    # remove offset of id and onset of id+1
    offset = np.delete(offset,id)
    onset = np.delete(onset,id+1)
    duration = offset-onset

    inter_bouts = [on_-off_ for (on_,off_) in zip(onset[1:],offset[:-1])]

    duration = offset-onset
    
    return onset,offset,duration,inter_bouts


def are_peaks_alernating(half_beat_pos,half_beat_neg):
    # if not a list: convert to list
    if isinstance(half_beat_pos, list):
        half_beat_pos = np.array(half_beat_pos)
    if isinstance(half_beat_neg, list):
        half_beat_neg = np.array(half_beat_neg)
    assert len(half_beat_pos) != 0,"half_beat_pos is empty."
    assert len(half_beat_neg) != 0,"half_beat_neg is empty."

    for x,y in zip(half_beat_pos[0:-1],half_beat_pos[1:]):
        if len(half_beat_neg[(half_beat_neg>x)&(half_beat_neg<y)])!=1:
            return False
    for x,y in zip(half_beat_neg[0:-1],half_beat_neg[1:]):
        if len(half_beat_pos[(half_beat_pos>x)&(half_beat_pos<y)])!=1:
            return False
    return True   

def clean_peak_forcing_alternation(half_beat_pos,half_beat_neg,tail):

    if isinstance(half_beat_pos, list):
        half_beat_pos = np.array(half_beat_pos)
    if isinstance(half_beat_neg, list):
        half_beat_neg = np.array(half_beat_neg)

    valid_half_beat_pos,valid_half_beat_neg = half_beat_pos.copy(),half_beat_neg.copy()

    b = are_peaks_alernating(valid_half_beat_pos,valid_half_beat_neg)==False
    while b:

        for x,y in zip(valid_half_beat_pos[0:-1],valid_half_beat_pos[1:]):
            if len(valid_half_beat_neg[(valid_half_beat_neg>x)&(valid_half_beat_neg<y)])==0: # Alternating peak in between
                # Remove the smallest value from the list
                if tail[x]>tail[y]:
                    valid_half_beat_pos = np.delete(valid_half_beat_pos, np.where(valid_half_beat_pos == y))
                    #valid_half_beat_pos.remove(y)
                else:
                    #valid_half_beat_pos.remove(x)
                    valid_half_beat_pos = np.delete(valid_half_beat_pos, np.where(valid_half_beat_pos == x))

                break

        for x,y in zip(valid_half_beat_neg[0:-1],valid_half_beat_neg[1:]):
            if len(valid_half_beat_pos[(valid_half_beat_pos>x)&(valid_half_beat_pos<y)])==0: # Alternating peak in between
                # Remove the smallest value from the list
                if tail[x]<tail[y]:
                    #valid_half_beat_neg.remove(y)
                    valid_half_beat_neg = np.delete(valid_half_beat_neg, np.where(valid_half_beat_neg == y))

                else:
                    #valid_half_beat_neg.remove(x)
                    valid_half_beat_neg = np.delete(valid_half_beat_neg, np.where(valid_half_beat_neg == x))

                break

        b = are_peaks_alernating(valid_half_beat_pos,valid_half_beat_neg)==False

    return valid_half_beat_pos,valid_half_beat_neg

def is_break_in_interbeatinterval(half_beat,Percentage=0.2,StartingIBI=5):

    ibi = [j-i for i,j in zip(half_beat[0:-1],half_beat[1:]) ]

    break_in_ibi = False
    i = 0
    if len(ibi)>StartingIBI:
        for k,(ibi_before,ibi_after) in enumerate(zip(ibi[StartingIBI-1:-1],ibi[StartingIBI:])):
            if (ibi_after/ibi_before>(1+Percentage)) or (ibi_after/ibi_before<(1-Percentage)):
                break_in_ibi = True
                i = half_beat[k+StartingIBI+1]
                break
                
    return break_in_ibi, i 

def is_break_in_amplitude(tail,all_peak,Percentage=0.2,StartingIBI=5):

    amplitude = [tail[f]-tail[l] for l,f in zip(all_peak[:-1],all_peak[1:])]
    break_in_amplitude = False
    i = 0
    if len(amplitude)>StartingIBI:
        for k,(a_before,a_after) in enumerate(zip(amplitude[StartingIBI:-1],amplitude[StartingIBI+1:])):
            ratio = np.abs(a_after/a_before)
            if (ratio>(1+Percentage)) or (ratio<(1-Percentage)):
                break_in_amplitude = True
                i = all_peak[k+StartingIBI+1]
                break
    return break_in_amplitude, i 
        
def interleave(peaks_pos,peaks_neg):
    
    if isinstance(peaks_pos, list)==False:
        peaks_pos = peaks_pos.tolist()
    if isinstance(peaks_neg, list)==False:
        peaks_neg = peaks_neg.tolist()        
    a = peaks_pos
    b = peaks_neg
    all_peak = a + b
    first_beat = min(peaks_pos+peaks_neg)
    sign_first = (1 if first_beat==min(peaks_pos) else -1)
    if sign_first>0:
        all_peak[::2] = a
        all_peak[1::2] = b
    else:
        all_peak[::2] = b
        all_peak[1::2] = a  

    return all_peak

def find_zeros_crossing(tail_speed):

    # Upward_zeros crossing
    id_up = np.where((tail_speed[0:-1]<0)&(tail_speed[1:]>0))[0] # Correspond to min  tail angle

    # Downward crossing
    id_down = np.where((tail_speed[0:-1]>0)&(tail_speed[1:]<0))[0] # Correspond to max tail angle


    return id_down,id_up

def are_peaks_distant(beat_pos,Distance):
    # if not a list: convert to list
    if isinstance(beat_pos, list):
        beat_pos = np.array(beat_pos)
    assert len(beat_pos) != 0,"beat_pos is empty."

    for x,y in zip(beat_pos[0:-1],beat_pos[1:]):
        if np.abs(y-x)<Distance:
            return False

    return True   

def clean_peak_forcing_distance(beat,tail,Distance=10,sign=1):

    if isinstance(beat, list):
        beat = np.array(beat)

    valid_beat = beat.copy()

    b = are_peaks_distant(valid_beat,Distance)==False
    while b:
        for x,y in zip(valid_beat[0:-1],valid_beat[1:]):
            if np.abs(y-x)<Distance: # Alternating peak in between
                # Remove the smallest value from the list
                if sign==1:
                        if tail[x]>tail[y]:
                            valid_beat = valid_beat[valid_beat!=y]
                        else:
                            valid_beat = valid_beat[valid_beat!=x]
                else:
                        if tail[x]<tail[y]:
                            valid_beat = valid_beat[valid_beat!=y]
                        else:
                            valid_beat = valid_beat[valid_beat!=x]
                break

        b = are_peaks_distant(valid_beat,Distance)==False

    return valid_beat

def is_there_oscillation(half_beat_pos,half_beat_neg):

    for x,y in zip(half_beat_pos[0:-1],half_beat_pos[1:]):
        if len(half_beat_neg[(half_beat_neg>x)&(half_beat_neg<y)])==1:
            return True

    for x,y in zip(half_beat_neg[0:-1],half_beat_neg[1:]):
        if len(half_beat_pos[(half_beat_pos>x)&(half_beat_pos<y)])==1:
            return True    
    return False
