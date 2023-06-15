

import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors


list_color_no_CS = ['#82cfff','#4589ff','#0000c8','#fcaf6d','#ffb3b8','#08bdba','#24a148','#9b82f3','#ee5396','#e3bc13','#fa4d56']
list_color_w_CS = ['#82cfff','#4589ff','#0000c8','#5d5d66','#000000','#fcaf6d','#ffb3b8','#08bdba','#24a148','#9b82f3','#ee5396','#e3bc13','#fa4d56']

NameCatSym_w_CS=['approach_swim', 'slow1', 'slow2', 'short_capture_swim','long_capture_swim', 'burst_swim', 'J_turn', 'high_angle_turn','routine_turn', 'spot_avoidance_turn', 'O_bend','long_latency_C_start', 'C_start']
NameCatSym_no_CS=['approach_swim', 'slow1', 'slow2', 'burst_swim', 'J_turn', 'high_angle_turn','routine_turn', 'spot_avoidance_turn', 'O_bend','long_latency_C_start', 'C_start']

NameCatShortSym_w_CS=['AS', 'S1', 'S2', 'SCS','LCS', 'BS', 'J', 'HAT','RT', 'SAT', 'O','LLC', 'SLC']
NameCatShortSym_no_CS=['AS', 'S1', 'S2', 'BS', 'J', 'HAT','RT', 'SAT', 'O','LLC', 'SLC']


cmp_bouts = colors.ListedColormap(list_color_w_CS)
cmp_bouts.set_under(color='white')
cmp_bouts.set_over(color='white')
cmp_bouts.set_bad(color='grey', alpha=None)


def display_trajectory(df,index,past_memory=3*700):
    #set up the figure
    fig = plt.figure(figsize=(5,5))
    canvas_width, canvas_height = fig.canvas.get_width_height()
    ax = fig.add_subplot()
    circle = plt.Circle((0, 0),25, ec='r', fill=False)
    ax.add_artist(circle)

    df_past = df.iloc[max(0,index-past_memory):index+1]
    ax.scatter(df_past['x'], df_past['y'], c = 'r', s =1,alpha=0.1)
    for i in range(0,df_past.shape[0],100):
        c = 1*np.cos(df_past.iloc[i]['angle'])
        s = 1*np.sin(df_past.iloc[i]['angle'])
        x_end = df_past.iloc[i]['x']
        y_end = df_past.iloc[i]['y']
        ax.arrow(x_end,y_end,c,s, head_width=1, head_length=1, fc='k', ec='k')

    c = 3*np.cos(df.iloc[index]['angle'])
    s = 3*np.sin(df.iloc[index]['angle'])
    x_end = df.iloc[index]['x']
    y_end = df.iloc[index]['y']
    ax.arrow(x_end,y_end,c,s, head_width=1, head_length=1, fc='b', ec='b')
    ax.set_xlim(-25,25)
    ax.set_ylim(-25,25)
    return fig

