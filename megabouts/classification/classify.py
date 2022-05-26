from sklearn.neighbors import KNeighborsClassifier
import numpy as np

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