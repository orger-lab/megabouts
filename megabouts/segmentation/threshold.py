import numpy as np
import scipy.signal as signal
from sklearn.mixture import GaussianMixture
from scipy import stats


def estimate_threshold_using_GMM(x,margin_std,axis=None):
    
    log_x = np.log(x[x>0])

    X = log_x[:,np.newaxis]
    gm = GaussianMixture(n_components=2, random_state=0).fit(X)

    weights = gm.weights_
    means = gm.means_
    covars = gm.covariances_

    id = np.argmin(means)
    sigma  = np.sqrt(covars[id]) # Standard Deviation
    BoutThresh = np.exp(means[id] + margin_std*sigma)[0]
    f_axis = log_x.copy().ravel()
    f_axis.sort()
    if axis is not None:
        axis.hist(log_x, bins=1000, histtype='bar', density=True, ec='red', alpha=0.1)
        #axis.plot(bins,count)
        axis.plot(f_axis,weights[0]*stats.norm.pdf(f_axis,means[0],np.sqrt(covars[0])).ravel(), c='blue')
        axis.plot(f_axis,weights[1]*stats.norm.pdf(f_axis,means[1],np.sqrt(covars[1])).ravel(), c='blue')
        axis.plot(f_axis,weights[0]*stats.norm.pdf(f_axis,means[0],np.sqrt(covars[0])).ravel()+weights[1]*stats.norm.pdf(f_axis,means[1],np.sqrt(covars[1])).ravel(),
        c='green')
        axis.scatter(means[id] + margin_std*sigma,0,s=100,c='r',marker=">")

    return BoutThresh[0],axis

