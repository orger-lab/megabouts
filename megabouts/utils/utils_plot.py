import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec


def display_canopy_member(df,All_Bouts,cat_list,filename_fig,title):
    n = int(np.ceil(np.sqrt(len(cat_list))))

    fig, ax = plt.subplots(facecolor='white',figsize=(15,15))
    plt.rc('font', family='monospace', serif='Courier New')
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('axes', labelsize=13)
    gs = fig.add_gridspec(n, n)

    for i in range(n):
        for j in range(n):
            ax = fig.add_subplot(gs[i,j])
            ax.set_ylim(-4,4)
            ax.set_xlim(0,100)
            ax.set_yticks([])
            ax.set_xticks([])
            try:
                k = int(cat_list[j+i*n])
                ax.plot(All_Bouts[df.all_label==k,:,8].T,'k',lw=0.3)
                ax.plot(All_Bouts[df.all_label_centroid==k,:,8].T,'r',lw=2)
                #ax.plot(All_Bouts_Centroid[k,:,8].T,'k',lw=0.3)
                #ax.plot(tail_angle_segment[id[::1],:,8].T,'k',lw=0.3)
                #ax.plot(tail_angle_segment[k,:,8],'r',lw=2)
                ax.set_xlim(0,100)
            except:
                pass

    plt.suptitle(title)
    plt.draw()
    plt.savefig(filename_fig)