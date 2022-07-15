
import numpy as np
import pandas as pd

from pybaselines.smooth import noise_median
from pybaselines.misc import beads
from pybaselines.whittaker import asls
from scipy.signal import savgol_filter

from functools import partial
'''
baseline_func = partial(compute_baseline_whittaker,
                        win_slow = 700,
                        win_std = int(700/2),
                        thresh_sigma = 1,
                        lmbda = 1e4)
baseline_func = partial(compute_baseline_beads,
                        win_slow = 700,
                        fc = 0.007,
                        lmbda = 1e1,
                        fps = 700)'''

def compute_baseline(x,method,params):
    """Main function to compute baseline

    Args:
        x (np.ndarray): input signal (T,)
        method (string): can be either 'slow' 'beads' or 'whittaker'
        params (dict): dict of parameters. Should contains fps field.
        If 'beads', should contain fields fc and lambda. If 'whittaker'
        should containd field lambda.

    Returns:
        np.ndarray: baseline of x
    """
    
    fps = params['fps']
 
    if method =='slow':
        baseline = noise_median(x, half_window=int(np.round(fps/2)), smooth_half_window=None, sigma=None)[0]
    elif method =='beads':
        baseline_func = partial(compute_baseline_beads,
                        win_slow = int(np.round(fps/2)),
                        fc = params['fc'],
                        lmbda = params['lmbda'],
                        fps = fps)
    elif method =='whittaker':
        baseline_func = partial(compute_baseline_whittaker,
                        win_slow = int(np.round(fps/2)),
                        win_std = int(fps/2),
                        thresh_sigma = 1,
                        lmbda = params['lmbda'])
    else:
        raise ValueError(f"Unsupported value for `method`: {method}")
    
    if method=='beads' or method=='whittaker':
        L_center = fps*28
        L_edge = fps
        x_batch,left_edge,center,right_edge,T,T_padded = compute_batch(x,L_center,L_edge)
        y_batch = compute_baseline_on_batch(x_batch,baseline_func)
        baseline = merge_batch(y_batch,T_padded,T,left_edge,center,right_edge,L_center,L_edge)
        # Further filtering to remove tiny oscillation:
        L_ms = 144
        L = int(np.ceil(L_ms*fps/1000))
        L = L+1 if L%2==0 else L
        baseline = savgol_filter(baseline,L, 3) 
        
    return baseline

    

def compute_baseline_whittaker(x,
                     win_slow = 700,
                     win_std = int(700/2),
                     thresh_sigma = 1,
                     lmbda = 1e4):
    
    slow_baseline = noise_median(x, half_window=win_slow, smooth_half_window=None, sigma=None)[0]
    x_centered = x-slow_baseline

    # Clipping values using std:
    std_ = pd.Series(x_centered).rolling(win_std).std().values
    mean_ = pd.Series(x_centered).rolling(win_std).mean().values
    upper_bound = mean_+thresh_sigma*std_
    lower_bound = mean_-thresh_sigma*std_
    sig = np.copy(x_centered)
    sig[sig>upper_bound]=upper_bound[upper_bound<sig]
    sig[sig<lower_bound]=lower_bound[lower_bound>sig]
    # Compute baseline of clipped signal:
    fast_baseline = asls(sig, lam=lmbda, p=0.5, diff_order=2, max_iter=50, tol=0.001, weights=None)[0]
    
    return slow_baseline+fast_baseline


def compute_baseline_beads(x,
                     win_slow = 700,
                     fc = 0.02,
                     lmbda = 1e4,
                     fps = 700):
    
    slow_baseline = noise_median(x, half_window=win_slow, smooth_half_window=None, sigma=None)[0]
    x_centered = x-slow_baseline

    fast_baseline = beads(x_centered, freq_cutoff=fc*700/fps,
                        lam_0=lmbda, lam_1=lmbda, lam_2=lmbda, asymmetry=1.0,
                        filter_type=1, cost_function='l1_v1', 
                        max_iter=50, tol=0.01, eps_0=1e-06, eps_1=1e-06, 
                        fit_parabola=False, smooth_half_window=3)[0
                                                                  ]
    return slow_baseline+fast_baseline


def compute_baseline_on_batch(x_batch,baseline_func):
    
    y_batch = []
    for x in x_batch:
        y_batch.append(baseline_func(x))
    
    return y_batch

def compute_batch(x,L_center,L_edge):
    """Batch input into overlapping segments. 
    The input will be mirror-padded to have a lenght divisible by L_center+L_edge.

    Args:
        x (np.ndarray): input time series to batch (T,)
        L_center (int): lenght of batch non overlapping  
        L_edge (int): lenght of overlap between batch, should be larger than 2

    Returns:
        - x_batch - list of batched input including padding
        - left_edge - list of interval for the left side overlapping segments of each batch
        - center - list of interval for the center of each batch
        - right edge - list of interval for the right side overlapping segments of each batch
        - T - length of input x
        - T_padded - lenght of padded input x
    """
    T = len(x)
    T_padded = int(np.ceil(T/(L_edge+L_center))*(L_center+L_edge))
    
    # Mirror x end to make it lenght T_padded
    x_padded = np.zeros(T_padded)
    x_padded[:len(x)] = x
    T_diff = len(x_padded)-len(x)
    x_padded[len(x)+1:] = x[:-T_diff:-1]

    # Batching:
    left_edge = [[0,0]]
    center = [[0,L_center]]
    right_edge = [[L_center,L_center+L_edge]]

    n = int(T_padded/(L_center+L_edge))
    for i in range(1,n):
        left_edge.append(right_edge[-1])
        center.append([right_edge[-1][1],right_edge[-1][1]+L_center])
        right_edge.append([center[-1][1],center[-1][1]+L_edge])

    # Compute batch:
    x_batch = []
    for i in range(len(center)):
        interval=(left_edge[i][0],right_edge[i][1])
        x_batch.append(x_padded[interval[0]:interval[1]])
        
    return x_batch,left_edge,center,right_edge,T,T_padded

def merge_batch(y_batch,T_padded,T,left_edge,center,right_edge,L_center,L_edge):
    """Unbatch input signal into a times series. 
    To avoid discontinuity, the overlapping segments of consecutive batches are smoothly averaged using a sigmoid weight.

    Args:
        y_batch (np.ndarray): list of batched input including padding
        T_padded (_type_): lenght of padded input 
        T (_type_): lenght of input before padding
        left_edge (_type_): list of interval for the left side overlapping segments of each batch
        center (_type_): list of interval for the center of each batch
        right_edge (_type_): list of interval for the right side overlapping segments of each batch
        L_center (_type_): lenght of batch non overlapping
        L_edge (_type_): lenght of overlap between batch

    Returns:
        y: merged signal 
    """
    x = np.linspace(-L_edge/2, L_edge/2, L_edge)
    sigma = L_edge/7
    z = 1/(1 + np.exp(-x/sigma))

    y = np.zeros(T_padded)
    # Center:
    y[center[0][0]:center[0][1]] = y_batch[0][0:L_center]
    # Right edge:
    val_1 = y_batch[0][L_center:]
    val_2 = y_batch[1][0:L_edge]
    val_merge = val_1*(1-z)+val_2*(z)
    y[right_edge[0][0]:right_edge[0][1]] = val_merge
    
    n = int(T_padded/(L_center+L_edge))
    for i in range(1,n-1):
        # Left:
        val_1 = y_batch[i-1][-L_edge:]
        val_2 = y_batch[i][0:L_edge]
        val_merge = val_1*(1-z)+val_2*(z)
        y[left_edge[i][0]:left_edge[i][1]] = val_merge
        # Center:
        y[center[i][0]:center[i][1]] = y_batch[i][L_edge:L_edge+L_center]
        # Right:
        val_1 = y_batch[i][L_center+L_edge:]
        val_2 = y_batch[i+1][0:L_edge]
        val_merge =val_1*(1-z)+val_2*(z)
        y[right_edge[i][0]:right_edge[i][1]] = val_merge

    # Left:
    val_1 = y_batch[-2][L_center+L_edge:]
    val_2 = y_batch[-1][:L_edge]
    val_merge = val_1*(1-z)+val_2*(z)
    y[left_edge[-1][0]:left_edge[-1][1]] = val_merge
    # Center
    y[center[-1][0]:right_edge[-1][1]] = y_batch[-1][L_edge:]
    
    y = y[0:T]
    
    return y
