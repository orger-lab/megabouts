import matplotlib.pyplot as plt
import h5py
from scipy.signal import find_peaks
import numpy as np

DEBUG_TEXT = False
DEBUG_PLOT = True
def debug(name,value=''):
    """prints variables and their contents when debuging"""
    if(DEBUG_TEXT):
        print(f'{name} : {value}')

def debug_plot( ax, 
                bout,bout_cumsum, 
                max_location_bout, min_location_bout,
                max_location_cumsum, min_location_cumsum,
                peak_search_start, peak_search_end, peak_location):
    """plots all relevat debug information"""
    if(ax != None):
        ax.plot(bout)
        ax.plot(bout_cumsum)
        # bout peaks
        ax.plot(max_location_bout , bout[max_location_bout],'+b')
        ax.plot(min_location_bout , bout[min_location_bout],'+r')
        # cumsum peaks
        ax.plot(max_location_cumsum , bout_cumsum[max_location_cumsum],'xb')
        ax.plot(min_location_cumsum , bout_cumsum[min_location_cumsum],'xr')
        # search range
        ax.plot([peak_search_start,peak_search_start] ,[-2,2],color=[1,0,1],linewidth=2)
        ax.plot([peak_search_end,peak_search_end] ,[-2,2],color=[0,1,1],linewidth=2)
        # selected peak
        if peak_location is np.NaN:
            ax.text(0,1,'ERROR',color=[1,1,1],backgroundcolor=[1,0,0],horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
        else:
            ax.plot(peak_location ,bout[peak_location],'og',ms = 10,mfc='none')

def align_bout_peaks(bout_data,quantile_threshold = 0.25 , minimum_peak_size = 0.25, minimum_peak_to_peak_amplitude = 4,debug_plot_axes=None):

    # only use the first 150 segments to calculate the alignment.
    bout = np.copy(bout_data[:min(len(bout_data),150)])
    # 1. Find peaks in data and remove peaks with small amplitude
    
    max_location_bout, _ = find_peaks(bout , height = minimum_peak_size)
    min_location_bout, _ = find_peaks(bout*-1, height = minimum_peak_size)
    peaks_bout = np.sort(np.concatenate((max_location_bout,min_location_bout)))
    debug('peak locations',peaks_bout)
    # 2. Smooth trace by zeroing small traces and calculate cumsum of the smoothed data
    tmp = bout
    threshold = np.quantile(abs(tmp), quantile_threshold)
    tmp[abs(tmp) < threshold] = 0
    bout_cumsum = np.cumsum(tmp)
    # 3. Find peaks in cumsum data
    max_location_cumsum, _ = find_peaks(bout_cumsum)
    min_location_cumsum, _ = find_peaks(bout_cumsum*-1)
    peaks_cumsum = np.sort(np.concatenate((max_location_cumsum,min_location_cumsum)))
    debug('peak locations cumsum',peaks_cumsum)
    peak_search_start = np.NaN
    peak_search_end = np.NaN
    # If there are peaks in the cumsum data, find the optimal place to search for the peak in the bout data
    if any(peaks_cumsum):
        # calculate the distances between consecutive peaks of the cumsum trace
        distances = np.abs(np.diff(np.pad(bout_cumsum[peaks_cumsum],(1,0))))
        # find first peak whose distance between peaks is bigger than the threshold
        idx = np.where(distances > minimum_peak_to_peak_amplitude)[0]
        
        debug('peak to peak amplitudes',distances)
        debug('first big peak index',idx)

        # skip the algorithm and throw error if there are no peaks in the cumsum data
        if idx.size != 0:
            idx = idx[0]
            # if there the first peak of the cumsum data is selected, check if there is any peak in the bout data before that peak
            if idx == 0:
                # if there is any peak in the bout data before the peak in the cumsum data,
                # set the search boundaries to be between the beginning of the data and the location of the peak in the cumsum data
                # if there aren't any peaks,
                # set the search boundaries to be between the location of the first and second peaks in the cumsum data
                debug('points before peak',np.where(peaks_bout < peaks_cumsum[0])[0])
                if len(np.where(peaks_bout < peaks_cumsum[0])[0]) > 0:
                    peak_search_start = 0
                    peak_search_end   = peaks_cumsum[0]
                else :
                    peak_search_start = peaks_cumsum[0]
                    peak_search_end   = peaks_cumsum[1]
            else:
                # if any other peak is selected, search between the selected point and the one before
                peak_search_start = peaks_cumsum[idx-1]
                peak_search_end   = peaks_cumsum[idx]
            debug('search start', peak_search_start)
            debug('search end',peak_search_end)
            # 4. Delete peaks outside the search range
            peaks_bout = peaks_bout[(peaks_bout > peak_search_start) & (peaks_bout < peak_search_end)]
        else:
            peaks_bout = []

    if any(peaks_bout):
        peak_location=peaks_bout[0]
        debug('selected peak',peak_location)
    else:
        peak_location = np.NaN
        #print('Failed to find a peak within these constraints')

    if(DEBUG_PLOT):
        debug_plot(debug_plot_axes,
                    bout,bout_cumsum, 
                    max_location_bout, min_location_bout,
                    max_location_cumsum, min_location_cumsum,
                    peak_search_start, peak_search_end, peak_location)
    return peak_location
