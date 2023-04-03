from megabouts.classification.template_bouts import Knn_Training_Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import numpy as np
from dataclasses import dataclass,field


@dataclass(repr=False)
class Classification():
    bout_category: np.ndarray = field(init=True)
    proba: np.ndarray = field(init=True)
    onset_shift: np.ndarray = field(init=True)
    outlier_score: np.ndarray = field(init=True)
    id_nearest_template: np.ndarray = field(init=True)
    id_nearest_template_aligned: np.ndarray = field(init=True)


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
    knn = KNeighborsClassifier(n_neighbors=n_neighbors,weights='distance')
    knn.fit(template_flat, kNN_training_dataset.labels)
    res = knn.kneighbors(X_flat)
    
    # Find delay and nearest template:
    id_nearest_template = res[1][:,0]
    d = kNN_training_dataset.delays[res[1]]
    onset_shift = stats.mode(d.T,keepdims=True).mode[0]
    
    # Compute aligned version of nearest template:
    i = id_nearest_template
    N = len(np.where(kNN_training_dataset.delays==0)[0])/2
    N_mid = len(kNN_training_dataset.delays)/2
    sg = [-1 if i>N_mid else 1 for i in id_nearest_template]
    mod_,div_ = np.divmod(id_nearest_template, N)
    id_nearest_template_aligned = np.array([int(i) if s>0 else int(i+N_mid) for i,s in zip(div_,sg)])

    # Compute Proba and Bout:
    res = knn.predict_proba(X_flat)
    proba = np.max(res,axis=1)
    bout_category = np.argmax(res,axis=1)

    # Compute Outlier:
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination='auto')
    lof.fit(template_flat)
    outlier_score = -lof.score_samples(X_flat)
        
    # Define Output:
    classification = Classification(bout_category=bout_category,
                                    proba=proba,
                                    onset_shift=onset_shift,
                                    outlier_score=outlier_score,
                                    id_nearest_template=id_nearest_template,
                                    id_nearest_template_aligned=id_nearest_template_aligned
                                    )

    return classification