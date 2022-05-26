import numpy as np

def find_onset_offset_numpy(binary_serie):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(binary_serie,1).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    onset = ranges[:,0]
    offset = ranges[:,1]
    duration = offset-onset
    return onset,offset,duration

from scipy.special import binom


def robust_diff(x,dt=1/700, filter_length=71):
    if not filter_length % 2 == 1:
        raise ValueError('Filter length must be odd.')
    M = int((filter_length - 1) / 2)
    m = int((filter_length - 3) / 2)
    coefs = [(1 / 2**(2 * m + 1)) * (binom(2 * m, m - k + 1) - binom(2 * m, m - k - 1))
        for k in range(1, M + 1)]
    coefs = np.array(coefs)
    kernel = np.concatenate((coefs[::-1],[0],-coefs))
    filtered = np.convolve(kernel,x, mode='valid')
    filtered = (1 / dt) * filtered
    filtered = np.concatenate((np.nan*np.ones(M),filtered,np.nan*np.ones(M)))
    return filtered

