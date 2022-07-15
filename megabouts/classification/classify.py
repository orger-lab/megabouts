from classification.template_bouts import Knn_Training_Dataset
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
import numpy as np


def array_normalizer(array:np.ndarray,
                     weight:np.ndarray,
                     )->np.ndarray:
    """Normalize each feature of the array according to the scale
    Args:
        array (np.ndarray): input of shape (n_samples, n_features, duration)
        weight (int): input array of shape (n_featues,)
    Returns:
        np.ndarray: scaled version of the input array
    """     
    assert len(weight)==array.shape[1]
    array_scaled = array*weight.reshape(1,len(weight),1)
    return array_scaled


#TODO: ONLY USE FLAY ARRAY WHEN REQUIRED

def bouts_classifier(X:np.ndarray,kNN_training_dataset:type(Knn_Training_Dataset),weight:np.ndarray,n_neighbors=5,tracking_method='tail_and_traj'):
    
    bout_duration = kNN_training_dataset.bout_duration
    if tracking_method=='tail_and_traj':
        template = kNN_training_dataset.tail_and_traj
        n_feat = kNN_training_dataset.tail_and_traj.shape[1]
    elif tracking_method=='tail':
        template = kNN_training_dataset.tail
        n_feat = kNN_training_dataset.tail.shape[1]
    elif tracking_method=='traj':
        template = kNN_training_dataset.traj
        n_feat = kNN_training_dataset.traj.shape[1]
    else:
        raise ValueError(f"Unsupported value for `tracking_method`: {tracking_method}")

    if len(weight)!=n_feat:
        raise ValueError(f"Size of weight should be {n_feat}")


    assert X.shape[1]==n_feat and X.shape[2]==bout_duration, \
        f"size of input should be: (n,{n_feat},{bout_duration}) to match the template"

    X_scaled = array_normalizer(X, weight)
    template_scaled = array_normalizer(template, weight)        
    # Apply Weighting to different features:
    X_flat = np.reshape(X_scaled,(X_scaled.shape[0],X_scaled.shape[1]*X_scaled.shape[2]))
    template_flat = np.reshape(template_scaled,(template_scaled.shape[0],template_scaled.shape[1]*template_scaled.shape[2]))
    
    # Compute Nearest Neigbor:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(template_flat, kNN_training_dataset.labels)
    res = knn.kneighbors(X_flat)

    id_nearest_template = res[1][:,0]
    l = kNN_training_dataset.labels[res[1]]
    d = kNN_training_dataset.delays[res[1]]

    m = stats.mode(l.T)
    bout_category = m.mode[0]

    id = np.where(m.count[0]==1)[0]
    # This make sure that if no class is overrepresented, the nearest example class is attributed
    if len(id)>0:
        bout_category[id] = l[id,0]
    #onset_delay = d[:,0]
    onset_delay_mode = stats.mode(d.T).mode[0]
    
    return bout_category,onset_delay_mode,id_nearest_template


