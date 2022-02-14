import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



list_color_no_CS = ['#82cfff','#4589ff','#0000c8','#fcaf6d','#ffb3b8','#08bdba','#24a148','#9b82f3','#ee5396','#e3bc13','#fa4d56']
list_color_w_CS = ['#82cfff','#4589ff','#0000c8','#5d5d66','#000000','#fcaf6d','#ffb3b8','#08bdba','#24a148','#9b82f3','#ee5396','#e3bc13','#fa4d56']
NameCatSym_w_CS=['approach_swim', 'slow1', 'slow2', 'slow_capture_swim','fast_capture_swim', 'burst_swim', 'J_turn', 'high_angle_turn','routine_turn', 'spot_avoidance_turn', 'O_bend','long_latency_C_start', 'C_start']
NameCatSym_no_CS=['approach_swim', 'slow1', 'slow2', 'burst_swim', 'J_turn', 'high_angle_turn','routine_turn', 'spot_avoidance_turn', 'O_bend','long_latency_C_start', 'C_start']

cmp = colors.ListedColormap(list_color_w_CS)


fig, ax = plt.subplots(figsize=(5,10))
for i,c in enumerate(list_color_w_CS):
    ax.add_patch(Rectangle(xy=(0,i*2), width=1,
                      height=1, facecolor=c))
    ax.text(1.1, i*2+0.5, NameCatSym_w_CS[i], fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')                     
ax.set_ylim(0,25)
ax.set_xlim(0,3)
ax.set_yticks(np.arange(0,26,2)+0.5)
ax.set_yticklabels(np.arange(13))
plt.show()

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

fig, ax = plt.subplots(figsize=(5,10))
for i,c in enumerate(list_color_no_CS):
    ax.add_patch(Rectangle(xy=(0,i*2), width=1,
                      height=1, facecolor=c))
    ax.text(1.1, i*2+0.5, NameCatSym_no_CS[i], fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')                     
ax.set_ylim(0,21)
ax.set_xlim(0,3)
ax.set_yticks(np.arange(0,22,2)+0.5)
ax.set_yticklabels(np.arange(11))
plt.show()