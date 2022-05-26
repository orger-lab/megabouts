from sporco.admm import cbpdnin
import numpy as np
from preprocessing.baseline import remove_slow_trend

def batch_tail_angle(tail_angle,batch_duration=700*30):
    N = int(np.ceil(tail_angle.shape[0]/(batch_duration)))
    Nseg = tail_angle.shape[1]
    tail_angle_ = np.zeros((N*batch_duration,Nseg))
    tail_angle_[:tail_angle.shape[0],:] = tail_angle
    tail_angle_batch =  tail_angle_.reshape(N,batch_duration,Nseg)
    tail_angle_batch = np.swapaxes(tail_angle_batch,0,1)
    tail_angle_batch = np.swapaxes(tail_angle_batch,1,2)
    return tail_angle_batch

def compute_sparse_code(tail_angle_detrend,Dict,Wg,lmbda=0.05,gamma=0.1,mu=0.5,Whn=60):


    # Batch Dataset:
    tail_angle_batch = batch_tail_angle(tail_angle_detrend)

    T_trial = tail_angle_batch.shape[0]
    N_Seg = tail_angle_batch.shape[1]
    N_atoms = Dict.shape[2]

    z = np.zeros((T_trial,N_atoms,1))

    opt = cbpdnin.ConvBPDNInhib.Options({'Verbose': True, 'MaxMainIter': 200,
                                            'RelStopTol': 5e-3, 'AuxVarObj': False,'HighMemSolve': True})                                    
    b = cbpdnin.ConvBPDNInhib(Dict[:,:,:],tail_angle_batch,lmbda=lmbda, Wg=Wg,gamma=gamma,mu=mu,Whn=Whn,win_args='box', opt=opt, dimK=1, dimN=1)
    z = b.solve().squeeze()
    tail_hat = b.reconstruct().squeeze()

    # Unbatch Result
    z_flat = np.zeros((z.shape[0]*z.shape[1],z.shape[2]))
    for i in range(z.shape[2]):
        z_flat[:,i] = z[:,:,i].T.flatten()
    z = np.copy(z_flat)
    z = z[:tail_angle_detrend.shape[0],:]
    tail_angle_hat_ = np.zeros((tail_hat.shape[0]*tail_hat.shape[2],tail_hat.shape[1]))
    for i in range(tail_hat.shape[1]):
        tail_angle_hat_[:,i] = tail_hat[:,i,:].T.flatten()
    tail_angle_hat_ = tail_angle_hat_[:tail_angle_detrend.shape[0],:]

    return z,tail_angle_hat_


def create_sparse_coder(Dict,lmbda=0.01,gamma=0.05,mu=0.05,Whn=60):
    
    def sparse_coder(tail_angle):
        N_atoms = Dict.shape[2]
        Wg = np.ones((1,N_atoms))
        # DETRENDING:
        tail_angle_detrend = remove_slow_trend(tail_angle,ref_segment=7)
        # SPARSE CODING:
        z,tail_angle_hat = compute_sparse_code(tail_angle_detrend[:,:7],Dict,Wg,lmbda=lmbda,gamma=gamma,mu=gamma,Whn=Whn)
        
        return z,tail_angle_hat
    
    return sparse_coder
