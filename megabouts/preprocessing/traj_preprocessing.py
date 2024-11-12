import numpy as np
import pandas as pd
from scipy.ndimage import shift
from ..utils.math_utils import robust_diff
from typing import Tuple
from ..config.preprocessing_config import TrajPreprocessingConfig


class TrajPreprocessingResult:
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
        """
        Preprocess trajectory data from a DataFrame.

        Args:
            traj_df (pd.DataFrame): DataFrame containing head trajectory data (x,y,yaw).

        Returns:
            pd.DataFrame: Preprocessed DataFrame with hierarchical structure.
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
        """Interpolates missing values in trajectory data."""
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
        """Apply 1€ filter over x."""
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
        """Compute the axial, lateral, and yaw speed of a body."""
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
        """Compute the kinematic activity of a body."""
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
