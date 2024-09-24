import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from ..config import BaseConfig
from ..utils import create_downsampling_function
from ..utils import create_hierarchical_df
from sporco.admm import cbpdnin


class SparseCodingResult:
    def __init__(self, tail_angle, sparse_code, decomposition, tail_angle_hat, regressor):
        self.tail_angle = tail_angle
        self.sparse_code = sparse_code
        self.decomposition = decomposition
        self.tail_angle_hat = tail_angle_hat
        self.regressor = regressor
        self.df = self._to_dataframe()

    def _to_dataframe(self):
        df_info = [
            ('tail_angle', 'segments', self.tail_angle),
            ('sparse_code', 'atoms', self.sparse_code),
            ('tail_angle_hat', 'segments', self.tail_angle_hat),
            ('decomposition', 'atoms', self.decomposition),
            ('regressor', 'atoms', self.regressor)
        ]
        return create_hierarchical_df(df_info)

class SparseCoding:
    """Class for sparse coding of tail angle data."""
    def __init__(self, config: SparseCodingConfig):
        self.config = config
        
    def sparse_code_tail_angle(self, tail_angle: np.ndarray) -> SparseCodingResult:
        """
        Sparse coding of tail angle data.

        Args:
            tail_angle (np.ndarray): array containing tail angle data.

        Returns:
            SparseCodingResult: Sparse Code
        """
        N_Seg = self.config.Dict.shape[1]
        T = tail_angle.shape[0]
        N_atoms = self.config.N_atoms

        tail_angle = tail_angle[:,:N_Seg]
        tail_angle_batch = self.batch_tail_angle(tail_angle)

        opt = cbpdnin.ConvBPDNInhib.Options({'Verbose': True, 
                                             'MaxMainIter': 200, 
                                             'RelStopTol': 5e-3,
                                             'AuxVarObj': False, 
                                             'HighMemSolve': True
                                             })
        
        b = cbpdnin.ConvBPDNInhib(
            self.config.Dict[:, :, :], tail_angle_batch,
            lmbda=self.config.lmbda, 
            Wg=np.ones((1, N_atoms)), gamma=self.config.gamma,
            mu=self.config.mu, Whn=self.config.window_inhib, win_args='box',
            opt=opt, dimK=1, dimN=1
        )
        z = b.solve().squeeze()
        z = SparseCoding.unbatch_result(z,T)
        tail_angle_hat = b.reconstruct().squeeze().swapaxes(1, 2)
        tail_angle_hat = SparseCoding.unbatch_result(tail_angle_hat,T)
        
        decomposition = self.compute_decomposition(z, tail_angle.shape[0])
        regressor = self.compute_vigor_decomposition(decomposition)
        return SparseCodingResult(tail_angle, z, decomposition, tail_angle_hat, regressor)

    @staticmethod
    def batch_tail_angle(tail_angle, batch_duration=20000):
        """
        Split a given tail angle sequence into batches of a given duration.

        Parameters:
        - tail_angle: 2D numpy array of tail angle values.
        - batch_duration: int, duration of each batch in number of time steps.

        Returns:
        - tail_angle_batch: 3D numpy array of tail angle values split into batches of the specified duration.
        """
        N = int(np.ceil(tail_angle.shape[0] / batch_duration))
        Nseg = tail_angle.shape[1]
        tail_angle_ = np.zeros((N * batch_duration, Nseg))
        tail_angle_[:tail_angle.shape[0], :] = tail_angle
        tail_angle_batch = tail_angle_.reshape(N, batch_duration, Nseg)
        return tail_angle_batch.swapaxes(0, 1).swapaxes(1, 2)
    
    @staticmethod
    def unbatch_result(result, original_length):
        """
        Unbatch the result back to its original shape.

        Parameters:
        - result: batched result array.
        - original_length: original length of the array.
        
        Returns:
        - Unbatched result array.
        """
        result_flat = np.zeros((result.shape[0]*result.shape[1],result.shape[2]))
        for i in range(result.shape[2]):
            result_flat[:,i] = result[:,:,i].T.flatten()

        return result_flat[:original_length,:]
    
    def compute_decomposition(self, z, original_length):
        """
        Compute the decomposition from the sparse codes.

        Parameters:
        - z: sparse codes array.
        - original_length: original length of the array.

        Returns:
        - Decomposition array.
        """
        decomposition = np.zeros((z.shape[1], original_length))
        for j in range(z.shape[1]):
            tmp = np.convolve(z[:, j], self.config.Dict[:, -1, j], 'full')
            decomposition[j, :] = tmp[:original_length]
        return decomposition.T

    def compute_vigor_decomposition(self,decomposition):
        regressor = np.zeros_like(decomposition)
        for k in range(decomposition.shape[1]):
            ts = pd.Series(decomposition[:,k])
            # calculate a rolling standard deviation
            regressor[:,k] = ts.rolling(window=self.config.vigor_win,center=True).std()
        return regressor