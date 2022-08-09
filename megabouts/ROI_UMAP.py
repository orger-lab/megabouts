from roipoly import RoiPoly
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
import pickle


filename = 'Free_Swim_Tail.pickle'
print(filename)

with open(filename, 'rb') as handle:
                u_embedding = pickle.load(handle)

embedding_list = u_embedding['embedding_list']
num_sub_clust = u_embedding['num_sub_clust']

sub_label_list = []
for i in range(len(embedding_list)):
    
    sub_label = np.zeros(embedding_list[i].shape[0])
    
    for s_c in range(num_sub_clust[i]-1):
        
        plt.figure(figsize=(10,10))
        plt.title(str(i)+' / '+ str(num_sub_clust[i]))
        for k in np.unique(sub_label):
            plt.scatter(embedding_list[i][sub_label==k,0],embedding_list[i][sub_label==k,1],s=5,alpha=0.1)
        my_roi = RoiPoly(color='r')
        verts = [(x,y) for x,y in zip(my_roi.x,my_roi.y)]
        path1 = Path(verts)
        index = path1.contains_points(embedding_list[i])
        sub_label[index] = s_c+1
    sub_label_list.append(sub_label)
    plt.show()


u_embedding={'embedding_list':embedding_list,'num_sub_clust':num_sub_clust,'sub_label_list':sub_label_list}

with open(filename, 'wb') as handle:
    pickle.dump(u_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)



