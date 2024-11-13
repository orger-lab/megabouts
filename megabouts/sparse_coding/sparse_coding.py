import numpy as np
import pandas as pd
from ..utils.data_utils import create_hierarchical_df

from sporco.admm import cbpdnin
from ..config.sparse_coding_config import SparseCodingConfig


class SparseCodingResult:
    def __init__(
        self,
        tail_angle: np.ndarray,
        sparse_code: np.ndarray,
        decomposition: np.ndarray,
        tail_angle_hat: np.ndarray,
        regressor: np.ndarray,
    ) -> None:
        self.tail_angle = tail_angle
        self.sparse_code = sparse_code
        self.decomposition = decomposition
        self.tail_angle_hat = tail_angle_hat
        self.regressor = regressor
        self.df = self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        """Convert results to a hierarchical DataFrame.

        Returns:
            pd.DataFrame: Hierarchical DataFrame containing all results
        """
        df_info = [
            ("tail_angle", "segments", self.tail_angle),
            ("sparse_code", "atoms", self.sparse_code),
            ("tail_angle_hat", "segments", self.tail_angle_hat),
            ("decomposition", "atoms", self.decomposition),
            ("regressor", "atoms", self.regressor),
        ]
        return create_hierarchical_df(df_info)


class SparseCoding:
    """Class for sparse coding of tail angle data."""

    def __init__(self, config: SparseCodingConfig):
        self.config = config

    def sparse_code_tail_angle(self, tail_angle: np.ndarray) -> SparseCodingResult:
        N_Seg = self.config.Dict.shape[1]
        T = tail_angle.shape[0]
        N_atoms = self.config.N_atoms

        tail_angle = tail_angle[:, :N_Seg]
        tail_angle_batch = self.batch_tail_angle(tail_angle)

        opt = cbpdnin.ConvBPDNInhib.Options(
            {
                "Verbose": True,
                "MaxMainIter": 200,
                "RelStopTol": 5e-3,
                "AuxVarObj": False,
                "HighMemSolve": True,
            }
        )

        b = cbpdnin.ConvBPDNInhib(
            self.config.Dict[:, :, :],
            tail_angle_batch,
            lmbda=self.config.lmbda,
            Wg=np.ones((1, N_atoms)),
            gamma=self.config.gamma,
            mu=self.config.mu,
            Whn=self.config.window_inhib,
            win_args="box",
            opt=opt,
            dimK=1,
            dimN=1,
        )
        z = b.solve().squeeze()
        z = SparseCoding.unbatch_result(z, T)
        tail_angle_hat = b.reconstruct().squeeze().swapaxes(1, 2)
        tail_angle_hat = SparseCoding.unbatch_result(tail_angle_hat, T)

        decomposition = self.compute_decomposition(z, tail_angle.shape[0])
        regressor = self.compute_vigor_decomposition(decomposition)
        return SparseCodingResult(
            tail_angle, z, decomposition, tail_angle_hat, regressor
        )

    @staticmethod
    def batch_tail_angle(
        tail_angle: np.ndarray, batch_duration: int = 20000
    ) -> np.ndarray:
        """Split a given tail angle sequence into batches.

        Args:
            tail_angle (np.ndarray): 2D array of tail angle values, shape (time_steps, n_segments)
            batch_duration (int, optional): Duration of each batch. Defaults to 20000.

        Returns:
            np.ndarray: 3D array of batched tail angles, shape (batch_duration, n_segments, n_batches)
        """
        N = int(np.ceil(tail_angle.shape[0] / batch_duration))
        Nseg = tail_angle.shape[1]
        tail_angle_ = np.zeros((N * batch_duration, Nseg))
        tail_angle_[: tail_angle.shape[0], :] = tail_angle
        tail_angle_batch = tail_angle_.reshape(N, batch_duration, Nseg)
        return tail_angle_batch.swapaxes(0, 1).swapaxes(1, 2)

    @staticmethod
    def unbatch_result(result: np.ndarray, original_length: int) -> np.ndarray:
        """Unbatch the result back to its original shape.

        Args:
            result (np.ndarray): Batched array of shape (time_steps, batch_size, n_features)
            original_length (int): Original sequence length to truncate to

        Returns:
            np.ndarray: Unbatched array of shape (original_length, n_features)
        """
        result_flat = np.zeros((result.shape[0] * result.shape[1], result.shape[2]))
        for i in range(result.shape[2]):
            result_flat[:, i] = result[:, :, i].T.flatten()

        return result_flat[:original_length, :]

    def compute_decomposition(self, z: np.ndarray, original_length: int) -> np.ndarray:
        """Compute the decomposition from the sparse codes.

        Args:
            z (np.ndarray): Sparse codes array of shape (time_steps, n_atoms)
            original_length (int): Original sequence length to truncate to

        Returns:
        - Decomposition array.
        """
        decomposition = np.zeros((z.shape[1], original_length))
        for j in range(z.shape[1]):
            tmp = np.convolve(z[:, j], self.config.Dict[:, -1, j], "full")
            decomposition[j, :] = tmp[:original_length]
        return decomposition.T

    def compute_vigor_decomposition(self, decomposition):
        regressor = np.zeros_like(decomposition)
        for k in range(decomposition.shape[1]):
            ts = pd.Series(decomposition[:, k])
            # calculate a rolling standard deviation
            regressor[:, k] = ts.rolling(
                window=self.config.vigor_win, center=True
            ).std()
        return regressor
