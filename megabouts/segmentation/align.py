import matplotlib.pyplot as plt
import h5py
from scipy.signal import find_peaks
import numpy as np

DEBUG_TEXT = False
DEBUG_PLOT = False
def debug(name,value=''):
    """prints variables and their contents when debuging"""
    if(DEBUG_TEXT):
        print(f'{name} : {value}')

def debug_plot( ax, 
                bout,bout_cumsum, 
                max_location_bout, min_location_bout,
                max_location_cumsum, min_location_cumsum,
                peak_search_start, peak_search_end, peak_location,
                quantile_threshold, minimum_peak_size, minimum_peak_to_peak_amplitude,merged_points):
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
        ax.plot(merged_points , bout_cumsum[merged_points],'Dk',fillstyle='none')
        # search range
        ax.plot([peak_search_start,peak_search_start] ,[-2,2],color=[1,0,1],linewidth=2)
        ax.plot([peak_search_end,peak_search_end] ,[-2,2],color=[0,1,1],linewidth=2)
        # selected peak
        if peak_location is np.NaN:
            ax.text(0,1,'ERROR',color=[1,1,1],backgroundcolor=[1,0,0],horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
        else:
            ax.plot(peak_location ,bout[peak_location],'og',ms = 10,mfc='none')

        plt.grid(color = 'green',which='both', linestyle = '--', linewidth = 0.5)

        # ax.text(0,0,f'quantile threshold {quantile_threshold}\n minimum peak size {minimum_peak_size}\n minimum peak to peak amplitude {minimum_peak_to_peak_amplitude}',color=[1,1,1],backgroundcolor=[0.88, 0.53, 0.26],horizontalalignment='left',verticalalignment='bottom', transform=ax.transAxes)


        
def align_bout_peaks(bout_data,quantile_threshold = 0.25 , minimum_peak_size = 0.25, min_peak_difference=0.30,debug_plot_axes=None):

    # only use the first 150 segments to calculate the alignment.
    bout = np.copy(bout_data[:min(len(bout_data),150)])
    # 1. Find peaks in data and remove peaks with small amplitude
    
    max_location_bout, _ = find_peaks(bout , height = minimum_peak_size)
    min_location_bout, _ = find_peaks(bout*-1, height = minimum_peak_size)
    peaks_bout = np.sort(np.concatenate((max_location_bout,min_location_bout)))

    debug('--------------- NORMAL DATA ---------------')

    debug('peak locations',peaks_bout)
    debug('peak values',bout[peaks_bout])
    # 2. Smooth trace by zeroing small traces and calculate cumsum of the smoothed data
    tmp = bout
    threshold = np.quantile(abs(tmp), quantile_threshold)
    tmp[abs(tmp) < threshold] = 0

 # bout_cumsum = np.cumsum(tmp)
    bout_cumsum = np.empty_like(tmp)
    bout_cumsum[0] = tmp[0]
    for i in range(1,len(bout_cumsum)):
        bout_cumsum[i] = bout_cumsum[i-1] + tmp[i] - (bout_cumsum[i-1]*0.05)


    # 3. Find peaks in cumsum data

    debug('--------------- CUMSUM DATA ---------------')


    max_location_cumsum, _ = find_peaks(bout_cumsum)
    min_location_cumsum, _ = find_peaks(bout_cumsum*-1)
    peaks_cumsum = np.sort(np.concatenate((max_location_cumsum,min_location_cumsum)))
    debug('peak locations cumsum',peaks_cumsum)
    debug('peak values cumsum',bout_cumsum[peaks_cumsum])


    if(DEBUG_PLOT):
        merged_points = np.copy(peaks_cumsum)

    values = bout_cumsum[peaks_cumsum]
    i = 0
    while i < len(values)-1 :
        v1 = values[i]
        v2 = values[i+1]

        if abs(v1-v2) < min_peak_difference :
            values = np.delete(values,i)
            peaks_cumsum = np.delete(peaks_cumsum,i)
            i -= 1
        i +=1

    debug('clean peak locations cumsum',peaks_cumsum)

    if(DEBUG_PLOT):
        merged_points = np.setxor1d(merged_points,peaks_cumsum)

    if len(peaks_cumsum) == 1 :
        peaks_cumsum = []


    #############################################

    peak_search_start = np.NaN
    peak_search_end = np.NaN
    # If there are peaks in the cumsum data, find the optimal place to search for the peak in the bout data
    if any(peaks_cumsum):
        debug('peaks found in cumsum data',len(peaks_cumsum))

        # calculate the distances between consecutive peaks of the cumsum trace
        distances = np.abs(np.diff(np.pad(bout_cumsum[peaks_cumsum],(1,0))))


        argmax = np.argmax(distances)


        m = np.min(distances)
        M = np.max(distances)
        l = M - m

        Tmin = 0
        Tmax = 1
        Tl = Tmax - Tmin

        normalized = (((distances - m) / l) * Tl) + Tmin 

        t = 0.75
        q = np.where(normalized[0:argmax+1] > t)[0]

        debug('norm',normalized)

        idx = q[0]


            
        if idx == 0:
            peak_search_start = 0
        else:
            peak_search_start = peaks_cumsum[idx-1]
        
        peak_search_end = peaks_cumsum[min(len(peaks_cumsum)-1,idx+1) ]


        debug('search start', peak_search_start)
        debug('search end',peak_search_end)

        # 4. Delete peaks outside the search range
        tmp = peaks_bout[(peaks_bout > peak_search_start) & (peaks_bout < peak_search_end)]
        if tmp.size > 0:
            peaks_bout = tmp
        else:
            peaks_bout = []
            debug('no peaks found in CURRENT interval')
            

    else:
        debug('No peaks found in cumsum data')



    if any(peaks_bout):

        debug('Peaks in the selected interval',peaks_bout)
        


        if len(peaks_cumsum) != 0:
            debug('there is more than 1 peak in normal bout AND 1st peak was selected in cumsum')

            x =  np.abs(bout[peaks_bout])
            debug('PEAKS',x)

            M = np.max(x)
            # print(M)

            mm = M * 0.85
            debug('SCALE',x/M)     
            debug('TRS',mm)

            #peak_location=peaks_bout[np.argmax(np.abs(bout[peaks_bout]))]
            pppp = np.where(x > mm)[0]
            debug("PKS",pppp )
            peak_location=peaks_bout[pppp[0]]
        else:
            debug('SELECTING 1ST PEAK')
            peak_location=peaks_bout[0]
        


        debug('--------------- RESULT ---------------', '')



        debug('selected peak',peak_location)
    else:
        peak_location = np.NaN
        debug('Failed to find a peak within these constraints',":")

    if(DEBUG_PLOT):
        debug_plot(debug_plot_axes,
                    bout,bout_cumsum, 
                    max_location_bout, min_location_bout,
                    max_location_cumsum, min_location_cumsum,
                    peak_search_start, peak_search_end, peak_location,
                    quantile_threshold, minimum_peak_size, minimum_peak_to_peak_amplitude,merged_points)
    return peak_location