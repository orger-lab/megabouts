from sporco.admm import cbpdnin
import numpy as np
from dataclasses import dataclass,field



@dataclass
class SparseCode():
    Dict: np.ndarray=field(init=True,repr=False)
    z: np.ndarray=field(init=True,repr=False)
    tail_angle_hat: np.ndarray=field(init=True,repr=False)
    decomposition: np.ndarray=field(init=True,repr=False)

    @property
    def n_atoms(self):
        return decomposition.shape[-1]
    
    

def batch_tail_angle(tail_angle,batch_duration=700*30):
    """
    Split a given tail angle sequence into batches of a given duration.

    Parameters:
    - tail_angle: 2D numpy array of tail angle values.
    - batch_duration: int, duration of each batch in number of time steps.

    Returns:
    - tail_angle_batch: 3D numpy array of tail angle values split into batches of the specified duration.
    """
    N = int(np.ceil(tail_angle.shape[0]/(batch_duration)))
    Nseg = tail_angle.shape[1]
    tail_angle_ = np.zeros((N*batch_duration,Nseg))
    tail_angle_[:tail_angle.shape[0],:] = tail_angle
    tail_angle_batch =  tail_angle_.reshape(N,batch_duration,Nseg)
    tail_angle_batch = np.swapaxes(tail_angle_batch,0,1)
    tail_angle_batch = np.swapaxes(tail_angle_batch,1,2)
    return tail_angle_batch


def compute_sparse_code(*,tail_angle,Dict,Wg=[],lmbda=0.05,gamma=0.1,mu=0.5,Whn=60):
    """
    Compute the sparse code of a given tail angle sequence using a dictionary of basis functions.

    Parameters:
    - tail_angle: 2D numpy array of tail angle values.
    - Dict: 3D numpy array of basis functions to use for the sparse code.
    - Wg: 1D numpy array of weights for the sparse code.
    - lmbda: float, regularization parameter to use in the sparse coding optimization.
    - gamma: float, weight to use for the inhibition term in the optimization.
    - mu: float, weight to use for the L1 norm in the optimization.
    - Whn: int, size of the convolutional window to use in the optimization.

    Returns:
    - sparse_code: instance of the SparseCode class, containing the computed sparse code and related information.
    """
    
    # Batch Dataset:
    tail_angle_batch = batch_tail_angle(tail_angle)

    T_trial = tail_angle_batch.shape[0]
    N_Seg = tail_angle_batch.shape[1]
    N_atoms = Dict.shape[2]

    z = np.zeros((T_trial,N_atoms,1))
    if not Wg:
        Wg = np.ones((1,N_atoms))
        
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
    z = z[:tail_angle.shape[0],:]
    tail_angle_hat_ = np.zeros((tail_hat.shape[0]*tail_hat.shape[2],tail_hat.shape[1]))
    for i in range(tail_hat.shape[1]):
        tail_angle_hat_[:,i] = tail_hat[:,i,:].T.flatten()
    tail_angle_hat_ = tail_angle_hat_[:tail_angle.shape[0],:]
    
    # Decomposition
    decomposition = np.zeros((z.shape[1],tail_angle.shape[0]))
    for j in range(z.shape[1]):
        tmp =  np.convolve(z[:,j],Dict[:,-1,j],'full')
        decomposition[j,:] = tmp[:tail_angle.shape[0]]
    decomposition = decomposition.T

    sparse_code = SparseCode(Dict=Dict,z=z,tail_angle_hat=tail_angle_hat_,decomposition=decomposition)
    return sparse_code
