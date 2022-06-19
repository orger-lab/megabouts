from megabouts.classification.template_bouts import Template
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
import numpy as np
'''
def create_classifier(templates_flat,templates_labels,n_neighbors=5):
    
    def classifier(bouts_array):
        
        # FLATTEN ARRAY:
        bouts_array = bouts_array[:,:,:]
        bouts_array_flat = np.reshape(np.swapaxes(bouts_array, 1, 2),(bouts_array.shape[0],bouts_array.shape[1]*bouts_array.shape[2]))

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(templates_flat, templates_labels)

        bout_cat = knn.predict(bouts_array_flat)

        return bout_cat

    return classifier'''
    
def array_normalizer(array:np.ndarray,
                     scale:int
                     )->np.ndarray:
    """Normalize each feature of the array according to the scale
    Args:
        array (np.ndarray): input of shape (n_samples, n_features, duration)
        scale (int): input array of shape (n_featues,)
    Returns:
        np.ndarray: scaled version of the input array
    """     
    assert len(scale)==array.shape[1]
    array_scaled = array*scale.reshape(-1,len(scale),-1)
    return array_scaled


#TODO: ONLY USE FLAY ARRAY WHEN REQUIRED

def create_classifier(template:Template,scale:np.ndarray,n_neighbors=5):
    
    def classifier(X:np.ndarray):
        
        assert X.shape[1]==template.bouts.shape[1] and X.shape[2]==template.bouts.shape[2], \
            "size of input does not match the template"
            
        ###### Scale Template and bouts: #####

        #templates_flat_normalized = array_normalizer(templates_flat, scale,Duration=Bout_Duration)
        #bouts_array_flat_normalized = array_normalizer(bouts_array_flat, scale,Duration=Bout_Duration)

        ##### Compute NN #####
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(templates_flat_normalized, templates_labels)
        res = knn.kneighbors(bouts_array_flat_normalized)

        id_nearest = res[1][:,0]
        l = templates_labels[res[1]]
        d = templates_delays[res[1]]

        m = stats.mode(l.T)
        label_pred = m.mode[0]

        id = np.where(m.count[0]==1)[0]
        # This make sure that if no class is overrepresented, the nearest example class is attributed
        if len(id)>0:
            label_pred[id] = l[id,0]
        delay_pred = d[:,0]

        return label_pred,delay_pred,id_nearest

    return classifier

