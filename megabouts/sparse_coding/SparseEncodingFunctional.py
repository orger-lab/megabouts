import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sporco.admm import cbpdn

from numba import njit, prange

def batch_tail_angle(tail_angle,batch_duration=700*60*2):
    N = int(np.ceil(tail_angle.shape[0]/(batch_duration)))
    Nseg = tail_angle.shape[1]
    tail_angle_ = np.zeros((N*batch_duration,Nseg))
    tail_angle_[:tail_angle.shape[0],:] = tail_angle
    tail_angle_batch =  tail_angle_.reshape(N,batch_duration,Nseg)
    tail_angle_batch = np.swapaxes(tail_angle_batch,0,1)
    tail_angle_batch = np.swapaxes(tail_angle_batch,1,2)
    return tail_angle_batch


def compute_sparse_code(dict_atoms,lambda_=1):

    def sparse_code(tail_angle):
        # Batch the tail angle:
        tail_angle_batch = batch_tail_angle(tail_angle,batch_duration=700*60*2)
        # Compute Sparse code
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 400,
                              'RelStopTol': 5e-3, 'AuxVarObj': False})

        b = cbpdn.ConvBPDN(dict_atoms,tail_angle_batch, lambda_, opt, dimN=1,dimK=1)
        z = b.solve().squeeze()
        tailhat = b.reconstruct().squeeze()

        # Unbatch the result
        z_flat = np.zeros((z.shape[0]*z.shape[1],z.shape[2]))
        for i in range(z.shape[2]):
            z_flat[:,i] = z[:,:,i].T.flatten()
        z = np.copy(z_flat)
        z = z[:tail_angle.shape[0],:]
        tail_angle_hat_ = np.zeros((tailhat.shape[0]*tailhat.shape[2],tailhat.shape[1]))
        for i in range(tailhat.shape[1]):
            tail_angle_hat_[:,i] = tailhat[:,i,:].T.flatten()
        tail_angle_hat_ = tail_angle_hat_[:tail_angle.shape[0],:]
        return z,tail_angle_hat_

    return sparse_code

def compute_likelihood_ratio_test(win_size=50):
    ### Math:
    # http://www.claudiobellei.com/2016/11/15/changepoint-frequentist/
    
    def likelihood_ratio_test_poisson(z):

        zpoisson = np.abs(z)
        # Make discrete:
        for n in range(zpoisson.shape[-1]):
            zpoisson[:,n] = pd.cut(zpoisson[:,n], bins=100, labels=False)
        
        num_time = z.shape[0]
        likelihood_ratio = np.zeros((num_time,zpoisson.shape[-1]))
        # Compute min lambda
        lambda_min=np.mean(zpoisson,axis=0)
        return sub_likelihood_loop(zpoisson,num_time,win_size,lambda_min)
        '''
        for t in prange(win_size,num_time-win_size):
            #for t in range(IdSt-500,IdSt+Duration+500):
            if (t%(700*60*5)==0):
                print(t/num_time)
            for n in range(zpoisson.shape[-1]):
                zfutur = zpoisson[t:t+win_size,n]
                zpast = zpoisson[t-win_size:t,n]
                zall = zpoisson[t-win_size:t+win_size,n]

                lambdaPast = max(np.mean(zpast),lambda_min[n])
                lambdaFuture = max(np.mean(zfutur),lambda_min[n])
                lambdaAll = max(np.mean(zall),lambda_min[n])

                likelihood_ratio[t,n] = np.sum(zpast)*np.log(lambdaPast) + np.sum(zfutur)*np.log(lambdaFuture) - np.sum(zall)*np.log(lambdaAll)

        return likelihood_ratio
        '''

    return likelihood_ratio_test_poisson

@njit(parallel=True)
def sub_likelihood_loop(zpoisson,num_time,win_size,lambda_min):
    likelihood_ratio = np.zeros((num_time,zpoisson.shape[-1]))
    for t in prange(win_size,num_time-win_size):
        for n in range(zpoisson.shape[-1]):
            zfutur = zpoisson[t:t+win_size,n]
            zpast = zpoisson[t-win_size:t,n]
            zall = zpoisson[t-win_size:t+win_size,n]

            lambdaPast = max(np.mean(zpast),lambda_min[n])
            lambdaFuture = max(np.mean(zfutur),lambda_min[n])
            lambdaAll = max(np.mean(zall),lambda_min[n])

            likelihood_ratio[t,n] = np.sum(zpast)*np.log(lambdaPast) + np.sum(zfutur)*np.log(lambdaFuture) - np.sum(zall)*np.log(lambdaAll)
    return likelihood_ratio