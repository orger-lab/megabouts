import numpy as np
import pandas as pd
from scipy.ndimage import shift
from ..utils.math_utils import robust_diff
from typing import Tuple
from ..config.preprocessing_config import TrajPreprocessingConfig


class TrajPreprocessingResult:
    """Container for trajectory preprocessing results.

    Parameters
    ----------
    x, y : np.ndarray
        Raw position coordinates
    yaw : np.ndarray
        Raw orientation angles
    x_smooth, y_smooth : np.ndarray
        Smoothed position coordinates
    yaw_smooth : np.ndarray
        Smoothed orientation angles
    axial_speed : np.ndarray
        Speed along body axis
    lateral_speed : np.ndarray
        Speed perpendicular to body axis
    yaw_speed : np.ndarray
        Angular velocity
    vigor : np.ndarray
        Kinematic activity measure
    no_tracking : np.ndarray
        Boolean mask indicating frames with no tracking
    """

    def __init__(
        self,
        x,
        y,
        yaw,
        x_smooth,
        y_smooth,
        yaw_smooth,
        axial_speed,
        lateral_speed,
        yaw_speed,
        vigor,
        no_tracking,
    ):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.x_smooth = x_smooth
        self.y_smooth = y_smooth
        self.yaw_smooth = yaw_smooth
        self.axial_speed = axial_speed
        self.lateral_speed = lateral_speed
        self.yaw_speed = yaw_speed
        self.vigor = vigor
        self.no_tracking = no_tracking
        self.df = self._to_dataframe()

    def _to_dataframe(self):
        columns = [
            "x",
            "y",
            "yaw",
            "x_smooth",
            "y_smooth",
            "yaw_smooth",
            "axial_speed",
            "lateral_speed",
            "yaw_speed",
            "vigor",
            "no_tracking",
        ]

        data = np.vstack(
            (
                self.x,
                self.y,
                self.yaw,
                self.x_smooth,
                self.y_smooth,
                self.yaw_smooth,
                self.axial_speed,
                self.lateral_speed,
                self.yaw_speed,
                self.vigor,
                self.no_tracking,
            )
        ).T

        return pd.DataFrame(data, index=range(data.shape[0]), columns=columns)


class TrajPreprocessing:
    """Class for preprocessing trajectory data."""

    def __init__(self, config: TrajPreprocessingConfig):
        self.config = config

    def preprocess_traj_df(self, traj_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess trajectory data from a DataFrame.

        Parameters
        ----------
        traj_df : pd.DataFrame
            DataFrame containing head trajectory data (x,y,yaw)

        Returns
        -------
        TrajPreprocessingResult
            Preprocessed trajectory data

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from megabouts.config import TrajPreprocessingConfig
        >>>
        >>> # Create sample data
        >>> t = np.linspace(0, 1, 100)  # 1 second at 100 fps
        >>> traj_df = pd.DataFrame({
        ...     'x': t,  # constant forward speed
        ...     'y': np.zeros_like(t),  # no lateral movement
        ...     'yaw': np.zeros_like(t)  # no rotation
        ... })
        >>>
        >>> # Preprocess
        >>> config = TrajPreprocessingConfig(fps=100)
        >>> preprocessor = TrajPreprocessing(config)
        >>> result = preprocessor.preprocess_traj_df(traj_df)
        >>>
        >>> # Verify forward movement was detected
        >>> np.mean(result.axial_speed[20:-20]) > 0  # positive forward speed
        True
        >>> np.allclose(result.lateral_speed[20:-20], 0, atol=1e-10)  # no lateral speed
        True
        """
        x_input, y_input, yaw_input = (
            traj_df["x"].values,
            traj_df["y"].values,
            traj_df["yaw"].values,
        )

        x, y, yaw, no_tracking = TrajPreprocessing.interp_traj_nan(
            x_input, y_input, yaw_input, limit_na=self.config.limit_na
        )

        def smooth_func(x):
            return TrajPreprocessing.one_euro_filter(
                x, self.config.freq_cutoff_min, self.config.beta, self.config.fps
            )

        x, y, yaw = map(smooth_func, [x, y, yaw])

        axial_speed, lateral_speed, yaw_speed = TrajPreprocessing.compute_speed(
            x, y, yaw, self.config.fps, n_diff=self.config.robust_diff
        )
        vigor = TrajPreprocessing.compute_kinematic_activity(
            axial_speed,
            lateral_speed,
            yaw_speed,
            lag=self.config.lag_kinematic_activity,
            fps=self.config.fps,
        )

        return TrajPreprocessingResult(
            x_input,
            y_input,
            yaw_input,
            x,
            y,
            yaw,
            axial_speed,
            lateral_speed,
            yaw_speed,
            vigor,
            no_tracking,
        )

    @staticmethod
    def interp_traj_nan(
        x: np.ndarray, y: np.ndarray, yaw: np.ndarray, limit_na: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Interpolates missing values in trajectory data.

        Parameters
        ----------
        x, y : np.ndarray
            Position coordinates
        yaw : np.ndarray
            Orientation angles
        limit_na : int
            Maximum number of consecutive NaN values to interpolate

        Returns
        -------
        x, y : np.ndarray
            Interpolated position coordinates
        yaw : np.ndarray
            Interpolated orientation angles
        no_tracking : np.ndarray
            Boolean mask indicating frames with no tracking
        """
        # Interpolate within limit na:
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
        yaw[~np.isnan(yaw)] = np.unwrap(yaw[~np.isnan(yaw)])

        def linear_fill_na(x):
            return pd.Series(x).interpolate(method="linear", limit=limit_na).values

        x, y, yaw = map(linear_fill_na, [x, y, yaw])
        no_tracking = np.logical_or.reduce((np.isnan(x), np.isnan(y), np.isnan(yaw)))

        # Interpolate remaining values:
        def linear_fill_na_no_limit(x):
            return (
                pd.Series(x).interpolate(method="linear", axis=0).ffill().bfill().values
            )

        x, y, yaw = map(linear_fill_na_no_limit, [x, y, yaw])

        return x, y, yaw, no_tracking

    @staticmethod
    def one_euro_filter(
        x: np.ndarray, freq_cutoff_min: float, beta: float, rate: int
    ) -> np.ndarray:
        """Apply 1€ filter over x.

        Parameters
        ----------
        x : np.ndarray
            Input signal
        freq_cutoff_min : float
            Minimum cutoff frequency in Hz
        beta : float
            Speed coefficient
        rate : int
            Sampling rate in Hz

        Returns
        -------
        np.ndarray
            Filtered signal
        """
        n_frames = len(x)
        x_smooth = np.zeros_like(x)
        x_smooth[0] = x[0]

        fc = freq_cutoff_min
        tau = 1 / (2 * np.pi * fc)
        te = 1 / rate
        alpha = 1 / (1 + tau / te)

        for i in range(1, n_frames):
            x_smooth[i] = alpha * x[i] + (1 - alpha) * x_smooth[i - 1]

            x_dot = (x_smooth[i] - x_smooth[i - 1]) * rate
            fc = freq_cutoff_min + beta * np.abs(x_dot)
            tau = 1 / (2 * np.pi * fc)
            te = 1 / rate
            alpha = 1 / (1 + tau / te)

        return x_smooth

    @staticmethod
    def compute_speed(
        x: np.ndarray, y: np.ndarray, yaw: np.ndarray, fps: int, n_diff: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the axial, lateral, and yaw speed of a body.

        Parameters
        ----------
        x, y : np.ndarray
            Position coordinates
        yaw : np.ndarray
            Orientation angles
        fps : int
            Frames per second
        n_diff : int
            Filter length for derivative computation

        Returns
        -------
        axial_speed : np.ndarray
            Speed along body axis
        lateral_speed : np.ndarray
            Speed perpendicular to body axis
        yaw_speed : np.ndarray
            Angular velocity

        Examples
        --------
        >>> t = np.linspace(0, 1, 100)  # 1 second at 100 fps
        >>> x = t  # constant forward speed
        >>> y = np.zeros_like(t)  # no lateral movement
        >>> yaw = np.zeros_like(t)  # no rotation
        >>> ax, lat, yaw_speed = TrajPreprocessing.compute_speed(x, y, yaw, fps=100, n_diff=11)
        >>> np.mean(ax[20:-20]) > 0  # positive forward speed
        True
        >>> np.allclose(lat[20:-20], 0, atol=1e-10)  # no lateral speed
        True
        """
        body_vector = np.array([np.cos(yaw), np.sin(yaw)])[:, :-1]
        position_change = np.zeros_like(body_vector)
        position_change[0, :] = robust_diff(x, dt=1 / fps, filter_length=n_diff)[:-1]
        position_change[1, :] = robust_diff(y, dt=1 / fps, filter_length=n_diff)[:-1]
        angle = np.pi / 2
        rotMat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        body_vector_orth = np.dot(rotMat, body_vector)

        axial_speed = np.einsum("ij,ij->j", body_vector, position_change)
        lateral_speed = np.einsum("ij,ij->j", body_vector_orth, position_change)
        yaw_speed = robust_diff(np.unwrap(yaw), dt=1 / fps, filter_length=n_diff)[:-1]
        axial_speed = np.concatenate((np.array([np.nan]), axial_speed))
        lateral_speed = np.concatenate((np.array([np.nan]), lateral_speed))
        yaw_speed = np.concatenate((np.array([np.nan]), yaw_speed))
        return axial_speed, lateral_speed, yaw_speed

    @staticmethod
    def compute_kinematic_activity(
        axial_speed: np.ndarray,
        lateral_speed: np.ndarray,
        yaw_speed: np.ndarray,
        lag: int,
        fps: int,
    ) -> np.ndarray:
        """Compute the kinematic activity of a body.

        Parameters
        ----------
        axial_speed : np.ndarray
            Speed along body axis
        lateral_speed : np.ndarray
            Speed perpendicular to body axis
        yaw_speed : np.ndarray
            Angular velocity
        lag : int
            Number of frames to compute activity over
        fps : int
            Frames per second

        Returns
        -------
        np.ndarray
            Kinematic activity measure

        Examples
        --------
        >>> n_frames = 100
        >>> speeds = np.zeros((n_frames, 3))  # no movement initially
        >>> speeds[50:60, 0] = 1  # burst of forward movement
        >>> activity = TrajPreprocessing.compute_kinematic_activity(
        ...     speeds[:, 0], speeds[:, 1], speeds[:, 2], lag=10, fps=100)
        >>> np.any(activity > 0)  # detected activity during burst
        True
        """
        traj_speed = np.vstack((axial_speed, lateral_speed, yaw_speed)).T
        traj_speed[np.isnan(traj_speed)] = 0
        traj_cumul = np.cumsum(np.abs(traj_speed), axis=0)

        dt = 1000 / fps * lag
        displacement = np.zeros_like(traj_cumul)
        for i in range(3):
            tmp = np.copy(traj_cumul[:, i])
            tmp_future_lag = shift(tmp, -lag, cval=0)
            displacement[:, i] = (tmp_future_lag - tmp) / dt

        displacement[:lag, :] = 0
        displacement[-lag:, :] = 0

        kinematic_activity = np.linalg.norm(displacement, axis=1)

        return kinematic_activity
