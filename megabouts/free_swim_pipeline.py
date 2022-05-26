from preprocessing.smoothing import clean_using_pca
from preprocessing.baseline import remove_slow_trend
from sparse_coding.sparse_coding import compute_sparse_code
from segmentation.align import align_bout_peaks
from scipy.signal import find_peaks

from classification.template_bouts import generate_template_bouts
from sklearn.neighbors import KNeighborsClassifier


def create_preprocess(limit_na=5,num_pcs=4):
    
    def preprocess(tail_angle) :
        # Interpolate NaN:
        for s in range(tail_angle.shape[1]):
            ds = pd.Series(tail_angle[:,s])
            ds.interpolate(method='nearest',limit=limit_na)
            tail_angle[:,s] = ds.values

        # Set to 0 for long sequence of nan:
        tail_angle[np.isnan(tail_angle)]=0

        # Use PCA for Cleaning (Could use DMD for better results)
        tail_angle = clean_using_pca(tail_angle,num_pcs=num_pcs)
        return tail_angle

    return preprocess

def create_sparse_coder(Dict,lmbda=0.01,gamma=0.05,mu=0.05,Whn=60):
    
    def sparse_coder(tail_angle):
        N_atoms = Dict.shape[2]
        Wg = np.ones((1,N_atoms))
        # DETRENDING:
        tail_angle_detrend = remove_slow_trend(tail_angle,ref_segment=7)
        # SPARSE CODING:
        z,tail_angle_hat = compute_sparse_code(tail_angle_detrend[:,:7],Dict,Wg,lmbda=lmbda,gamma=gamma,mu=gamma,Whn=Whn)
        
        return z,tail_angle_hat
    
    return sparse_coder


def create_segmentation_from_code(Min_Code_Ampl=1,SpikeDist=120,Bout_Duration=140):

    def segment_from_code(z,tail_angle):
    
        # FINDING PEAKS IN SPARSE CODE:
        z_max = np.max(np.abs(z),axis=1)
        peaks, _ = find_peaks(z_max, height=Min_Code_Ampl,distance=SpikeDist)
        peaks_bin = np.zeros(tail_angle.shape[0])
        peaks_bin[peaks]=1

        '''kernel = np.ones(SpikeDist)
        filtered_forward = np.convolve(kernel,peaks_bin, mode='full')[:peaks_bin.shape[0]]
        is_tail_active = 1.0*(filtered_forward>0)'''

        # EXTRACT BOUTS:
        bouts_array = np.zeros((len(peaks),Bout_Duration,7))
        bouts_hat_array = np.zeros((len(peaks),Bout_Duration,7))

        onset = []
        offset = []
        aligned_peaks = []

        i = 0
        for iter_,peak in enumerate(peaks):
                if ((peak>20)&(peak+120<T)):
                    id_st = peak - 20
                    id_ed = id_st +140
                    tmp = tail_angle[id_st:id_ed,7]
                    peak_location = align_bout_peaks(tmp,quantile_threshold = 0.25 , minimum_peak_size = 0.25, minimum_peak_to_peak_amplitude = 4,debug_plot_axes=None)
                    if np.isnan(peak_location):
                        peak_location = peak
                    else:
                        aligned_peaks.append(id_st+peak_location)
                        id_st = id_st+peak_location - 40
                        id_ed = id_st +140
                    bouts_array[i,:,:] = tail_angle[id_st:id_ed,:7]
                    i = i+1
                    onset.append(id_st)
                    offset.append(id_ed)
        bouts_array = bouts_array[:i,:,:]
        return onset,offset,bouts_array

    return segment_from_code

def create_classifier(templates_flat,templates_labels,n_neighbors=5):
    
    def classifier(bouts_array):
        
        # FLATTEN ARRAY:
        bouts_array = bouts_array[:,:,:]
        bouts_array_flat = np.reshape(np.swapaxes(bouts_array, 1, 2),(bouts_array.shape[0],bouts_array.shape[1]*bouts_array.shape[2]))

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(templates_flat, templates_labels)

        bout_cat = knn.predict(bouts_array_flat)

        return bout_cat

    return classifier
