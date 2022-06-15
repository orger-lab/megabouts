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

def array_normalizer(array,scale,Duration=140):
    N = int(len(scale))
    X = array.reshape(-1,N,Duration)
    for i in range(N):
        X[:,i,:] = X[:,i,:]*scale[i]
    X_flat =   X.reshape(-1,Duration*N)
    return X_flat


def create_classifier(templates_flat,templates_labels,templates_delays,scale,n_neighbors=5,Bout_Duration=140):
    
    def classifier(bouts_array_flat):
        
        ###### Scale Template and bouts: #####

        #scale_x,scale_y,scale_theta = 1/np.std(templates_flat[:,:Bout_Duration]),1/np.std(templates_flat[:,Bout_Duration:Bout_Duration*2]),1/np.std(templates_flat[:,Bout_Duration*2:])
        #scale_x,scale_y,scale_theta = 0.5,0.4,1
        #scale_tail = 1.6
        '''
        templates_flat_normalized = np.copy(templates_flat)
        templates_flat_normalized[:,:Bout_Duration] = templates_flat_normalized[:,:Bout_Duration]*scale_x
        templates_flat_normalized[:,Bout_Duration:Bout_Duration*2] = templates_flat_normalized[:,Bout_Duration:Bout_Duration*2]*scale_y
        templates_flat_normalized[:,Bout_Duration*2:] = templates_flat_normalized[:,Bout_Duration*2:]*scale_theta

        bouts_array_flat_normalized = np.copy(bouts_array_flat)
        bouts_array_flat_normalized[:,:Bout_Duration] = bouts_array_flat_normalized[:,:Bout_Duration]*scale_x
        bouts_array_flat_normalized[:,Bout_Duration:Bout_Duration*2] = bouts_array_flat_normalized[:,Bout_Duration:Bout_Duration*2]*scale_y
        bouts_array_flat_normalized[:,Bout_Duration*2:] = bouts_array_flat_normalized[:,Bout_Duration*2:]*scale_theta
        '''
        templates_flat_normalized = array_normalizer(templates_flat, scale,Duration=Bout_Duration)
        bouts_array_flat_normalized = array_normalizer(bouts_array_flat, scale,Duration=Bout_Duration)

        ##### Compute NN #####
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(templates_flat_normalized, templates_labels)
        res = knn.kneighbors(bouts_array_flat_normalized)

        '''
        Nearest_bouts_x = templates_flat_normalized[res[1][:,0],:140]
        Nearest_bouts_y = templates_flat_normalized[res[1][:,0],140:140*2]
        Nearest_bouts_angle = templates_flat_normalized[res[1][:,0],140*2:]
        '''
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

