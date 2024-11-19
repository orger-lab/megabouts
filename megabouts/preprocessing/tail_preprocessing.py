from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, convolve
from scipy.signal.windows import boxcar
from sklearn.decomposition import PCA

from ..config.preprocessing_config import TailPreprocessingConfig
from ..preprocessing.tail_baseline import compute_baseline
from ..utils.math_utils import robust_diff
from ..utils.data_utils import create_hierarchical_df


class TailPreprocessingResult:
    """Container for tail preprocessing results.

    Parameters
    ----------
    angle : np.ndarray
        Raw tail angles
    angle_baseline : np.ndarray
        Computed baseline of tail angles
    angle_smooth : np.ndarray
        Smoothed and baseline-substracted tail angles
    vigor : np.ndarray
        Computed tail vigor
    no_tracking : np.ndarray
        Boolean mask indicating frames with no tracking
    """

    def __init__(self, angle, angle_baseline, angle_smooth, vigor, no_tracking):
        self.angle = angle
        self.angle_baseline = angle_baseline
        self.angle_smooth = angle_smooth
        self.vigor = vigor
        self.no_tracking = no_tracking
        self.df = self._to_dataframe()

    def _to_dataframe(self):
        df_info = [
            ("angle", "segments", self.angle),
            ("angle_baseline", "segments", self.angle_baseline),
            ("angle_smooth", "segments", self.angle_smooth),
            ("vigor", "None", self.vigor),
            ("no_tracking", "None", self.no_tracking),
        ]
        df = create_hierarchical_df(df_info)
        return df


class TailPreprocessing:
    """Class for preprocessing tail angle data."""

    def __init__(self, config: TailPreprocessingConfig):
        self.config = config

    def preprocess_tail_df(self, tail_df: pd.DataFrame) -> TailPreprocessingResult:
        """Preprocess tail angle data from a DataFrame.

        Parameters
        ----------
        tail_df : pd.DataFrame
            DataFrame containing tail angle data

        Returns
        -------
        TailPreprocessingResult
            Preprocessed tail data

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from megabouts.config import TailPreprocessingConfig
        >>> # Create sample tail angles (100 timepoints, 10 segments)
        >>> tail_df = pd.DataFrame(
        ...     np.sin(np.linspace(0, 2*np.pi, 100))[:, None] * np.ones((100, 10)),
        ...     columns=[f'angle_{i}' for i in range(10)]
        ... )
        >>> config = TailPreprocessingConfig(fps=100)
        >>> result = TailPreprocessing(config).preprocess_tail_df(tail_df)
        >>> result.angle.shape == (100, 10)
        True
        >>> result.angle.shape == result.angle_smooth.shape
        True
        >>> result.vigor.ndim == 1  # vigor is 1D time series
        True
        """
        # Extract Tail Angle
        angle_input = tail_df[["angle_" + str(i) for i in range(10)]].values
        # Smoothing
        angle, angle_baseline, no_tracking = self.preprocess_tail_angle(
            angle=angle_input
        )
        angle -= angle_baseline

        vigor = TailPreprocessing.compute_tail_speed(
            angle=angle,
            fps=self.config.fps,
            tail_speed_filter=self.config.tail_speed_filter,
            tail_speed_boxcar_filter=self.config.tail_speed_boxcar_filter,
        )

        return TailPreprocessingResult(
            angle_input, angle_baseline, angle, vigor, no_tracking
        )

    def preprocess_tail_angle(self, angle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess raw tail angle data.

        Parameters
        ----------
        angle : np.ndarray
            Raw tail angles, shape (T, n_segments)

        Returns
        -------
        angle : np.ndarray
            Preprocessed angles
        angle_baseline : np.ndarray
            Computed baseline
        no_tracking : np.ndarray
            Boolean mask for frames with no tracking
        """
        angle, no_tracking = self.interp_tail_nan(angle, limit_na=self.config.limit_na)

        angle = self.clean_using_pca(angle, num_pcs=self.config.num_pcs)

        angle = self.smooth_tail_angle(angle, savgol_window=self.config.savgol_window)

        angle_baseline = self.compute_baseline(
            angle,
            baseline_method=self.config.baseline_method,
            baseline_params=self.config.baseline_params,
        )

        return angle, angle_baseline, no_tracking

    @staticmethod
    def interp_tail_nan(
        angle: np.ndarray, limit_na: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolates missing values in tail angle.

        Parameters
        ----------
        angle : np.ndarray
            Tail angles with potential NaN values, shape (T, n_segments)
        limit_na : int, optional
            Maximum number of consecutive NaN values to interpolate, by default 5

        Returns
        -------
        angle_no_nan : np.ndarray
            Interpolated angles
        no_tracking : np.ndarray
            Boolean mask indicating frames with no tracking
        """
        # Interpolate NaN timestep:
        angle_no_nan = np.zeros_like(angle)
        for s in range(angle.shape[1]):
            ds = pd.Series(angle[:, s])
            ds.interpolate(method="nearest", limit=limit_na, inplace=True)
            angle_no_nan[:, s] = ds.values

        no_tracking = np.isnan(np.sum(angle_no_nan, axis=1))
        angle_no_nan[np.isnan(angle_no_nan)] = 0

        return angle_no_nan, no_tracking

    @staticmethod
    def clean_using_pca(angle: np.ndarray, num_pcs=4) -> np.ndarray:
        """Apply PCA autoencoding to clean up a multidimensional time series.

        Parameters
        ----------
        angle : np.ndarray
            Input angles, shape (T, n_segments)
        num_pcs : int, optional
            Number of principal components to use, by default 4

        Returns
        -------
        np.ndarray
            PCA-cleaned angles
        """
        pca = PCA(n_components=num_pcs)
        pca.fit(angle)
        low_D = pca.transform(angle)
        angle_hat = pca.inverse_transform(low_D)
        return angle_hat

    @staticmethod
    def smooth_tail_angle(angle: np.ndarray, savgol_window: int) -> np.ndarray:
        """Smooth the tail angle data using Savitzky-Golay filter.

        Parameters
        ----------
        angle : np.ndarray
            Input angles, shape (T, n_segments)
        savgol_window : int
            Window length for Savitzky-Golay filter (must be odd)

        Returns
        -------
        np.ndarray
            Smoothed angles
        """
        angle_smooth = np.copy(angle)
        if savgol_window > 2:
            for n in range(angle.shape[1]):
                angle_smooth[:, n] = savgol_filter(
                    angle[:, n],
                    savgol_window,
                    2,
                    deriv=0,
                    delta=1.0,
                    axis=-1,
                    mode="interp",
                    cval=0.0,
                )
        return angle_smooth

    @staticmethod
    def compute_baseline(
        angle_smooth: np.ndarray, baseline_method: str, baseline_params: dict
    ) -> np.ndarray:
        """Compute the baseline for the smoothed tail angle data.

        Parameters
        ----------
        angle_smooth : np.ndarray
            Smoothed angles, shape (T, n_segments)
        baseline_method : str
            Method for baseline computation
        baseline_params : dict
            Parameters for baseline computation

        Returns
        -------
        np.ndarray
            Computed baseline
        """
        angle_baseline = np.zeros_like(angle_smooth)
        for s in range(angle_smooth.shape[1]):
            angle_baseline[:, s] = compute_baseline(
                angle_smooth[:, s], baseline_method, baseline_params
            )
        return angle_baseline

    @staticmethod
    def compute_tail_speed(
        angle: np.ndarray,
        fps: int,
        tail_speed_filter: int,
        tail_speed_boxcar_filter: int,
    ) -> np.ndarray:
        """Compute tail speed and vigor.

        Parameters
        ----------
        angle : np.ndarray
            Input angles, shape (T, n_segments)
        fps : int
            Frames per second
        tail_speed_filter : int
            Filter length for speed computation
        tail_speed_boxcar_filter : int
            Filter length for boxcar smoothing

        Returns
        -------
        np.ndarray
            Computed tail vigor
        """
        angle_speed = np.zeros_like(angle)
        for i in range(angle.shape[1]):
            angle_speed[:, i] = robust_diff(
                angle[:, i], dt=1 / fps, filter_length=tail_speed_filter
            )
        cumul_filtered_speed = np.sum(np.abs(angle_speed), axis=1)
        vigor = convolve(
            cumul_filtered_speed,
            boxcar(tail_speed_boxcar_filter) / tail_speed_boxcar_filter,
            mode="same",
        )
        return vigor
