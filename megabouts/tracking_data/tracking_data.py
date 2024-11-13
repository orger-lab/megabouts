from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from .convert_tracking import compute_angles_from_keypoints
from .convert_tracking import convert_tail_angle_to_keypoints
from .convert_tracking import interpolate_tail_keypoint, interpolate_tail_angle


class TrackingConfig:
    """Configuration for zebrafish tracking datasets.

    Args:
        fps (int): Frames per second.
        tracking (str): Type of tracking ('full_tracking', 'head_tracking', 'tail_tracking').

    Raises:
        AttributeError: If invalid tracking type or input type is provided.

    Example:
        >>> from megabouts.tracking_data.data_loader import TrackingConfig
        >>> tracking_config = TrackingConfig(fps=30, tracking='full_tracking')
        >>> print(tracking_config.fps)
        30
        >>> print(tracking_config.tracking)
        full_tracking
    """

    def __init__(self, *, fps, tracking):
        tracking_options = ["full_tracking", "head_tracking", "tail_tracking"]
        if tracking not in tracking_options:
            raise AttributeError(f"tracking should be among {tracking_options}")

        if not (20 <= fps <= 700) or not (fps == np.round(fps)):
            raise AttributeError("fps should be an integer between 20 and 700")

        self.fps = int(fps)
        self.tracking = tracking


class TrackingData(ABC):
    @classmethod
    @abstractmethod
    def from_keypoints(*args):
        raise NotImplementedError("This method should be overridden by subclasses")

    @classmethod
    @abstractmethod
    def from_posture(*args):
        raise NotImplementedError("This method should be overridden by subclasses")

    @staticmethod
    @abstractmethod
    def _validate_keypoints(*args):
        raise NotImplementedError("This method should be overridden by subclasses")

    @staticmethod
    @abstractmethod
    def _validate_posture(*args):
        raise NotImplementedError("This method should be overridden by subclasses")


class FullTrackingData(TrackingData):
    def __init__(self, head_x, head_y, head_yaw, tail_x, tail_y, tail_angle):
        self._tail_x = tail_x
        self._tail_y = tail_y
        self._tail_angle = tail_angle
        self._head_x = head_x
        self._head_y = head_y
        self._head_yaw = head_yaw
        self.T = len(self._tail_angle)

    @classmethod
    def from_keypoints(cls, *, head_x, head_y, tail_x, tail_y):
        cls._validate_keypoints(head_x, head_y, tail_x, tail_y)
        if tail_x.shape[1] != 11:
            tail_x, tail_y = interpolate_tail_keypoint(tail_x, tail_y, 10)
        tail_angle, head_yaw = compute_angles_from_keypoints(
            head_x, head_y, tail_x, tail_y
        )
        return cls(head_x, head_y, head_yaw, tail_x, tail_y, tail_angle)

    @classmethod
    def from_posture(cls, *, head_x, head_y, head_yaw, tail_angle):
        cls._validate_posture(head_x, head_y, head_yaw, tail_angle)
        if tail_angle.shape[1] != 10:
            tail_angle = interpolate_tail_angle(tail_angle, 10)
        #
        #
        return cls(head_x, head_y, head_yaw, None, None, tail_angle)

    @property
    def tail_df(self):
        tail_df = pd.DataFrame(
            self._tail_angle, columns=[f"angle_{i}" for i in range(10)]
        )
        return tail_df

    @property
    def traj_df(self):
        traj_df = pd.DataFrame(
            {"x": self._head_x, "y": self._head_y, "yaw": self._head_yaw}
        )
        return traj_df

    @property
    def tail_keypoints_df(self):
        if self._tail_x is None or self._tail_y is None:
            self._tail_x, self._tail_y = convert_tail_angle_to_keypoints(
                self._head_x,
                self._head_y,
                self._head_yaw,
                self._tail_angle,
                body_to_tail_mm=0.5,
                tail_to_tail_mm=0.32,
            )
        tail_keypoints_df = pd.DataFrame(
            {"tail_x": self._tail_x, "tail_y": self._tail_y}
        )
        return tail_keypoints_df

    @staticmethod
    def _validate_keypoints(head_x, head_y, tail_x, tail_y):
        T = len(head_x)
        if not (len(head_y) == T and tail_x.shape[0] == T and tail_y.shape[0] == T):
            raise ValueError("All inputs must have the same number of time points (T).")
        N_keypoints = tail_x.shape[1]
        if N_keypoints < 4:
            raise ValueError(
                "At least 4 points from swim bladder to tail tips are required for full tracking"
            )
        if tail_x.shape[1] != tail_y.shape[1]:
            raise ValueError(
                "tail_x and tail_y must have the same number of keypoints (N)."
            )

    @staticmethod
    def _validate_posture(head_x, head_y, head_yaw, tail_angle):
        T = len(head_x)
        if not (
            len(head_y) == T and head_yaw.shape[0] == T and tail_angle.shape[0] == T
        ):
            raise ValueError("All inputs must have the same number of time points (T).")
        N_keypoints = tail_angle.shape[1] + 1
        if N_keypoints < 4:
            raise ValueError(
                "At least 4 points from swim bladder to tail tips are required for full tracking"
            )


class HeadTrackingData(TrackingData):
    def __init__(self, head_x, head_y, head_yaw, swimbladder_x, swimbladder_y):
        self._head_x = head_x
        self._head_y = head_y
        self._head_yaw = head_yaw
        self._swimbladder_x = swimbladder_x
        self._swimbladder_y = swimbladder_y
        self.T = len(self._head_x)

    @classmethod
    def from_keypoints(cls, *, head_x, head_y, swimbladder_x, swimbladder_y):
        cls._validate_keypoints(head_x, head_y, swimbladder_x, swimbladder_y)
        tail_angle, head_yaw = compute_angles_from_keypoints(
            head_x, head_y, swimbladder_x[:, np.newaxis], swimbladder_y[:, np.newaxis]
        )
        return cls(head_x, head_y, head_yaw, swimbladder_x, swimbladder_y)

    @classmethod
    def from_posture(cls, *, head_x, head_y, head_yaw):
        cls._validate_posture(head_x, head_y, head_yaw)
        # tail_angle = np.zeros((len(head_x),10))
        # tail_x,tail_y = convert_tail_angle_to_keypoints(head_x, head_y, head_yaw, tail_angle, body_to_tail_mm=0.5, tail_to_tail_mm=0.32)
        # swimbladder_x,swimbladder_y = tail_x[:,0],tail_y[:,0]
        # return cls(head_x,head_y,head_yaw,swimbladder_x,swimbladder_y)
        return cls(head_x, head_y, head_yaw, None, None)

    @property
    def traj_df(self):
        traj_df = pd.DataFrame(
            {"x": self._head_x, "y": self._head_y, "yaw": self._head_yaw}
        )
        return traj_df

    @staticmethod
    def _validate_keypoints(head_x, head_y, swimbladder_x, swimbladder_y):
        T = len(head_x)
        if not (
            len(head_y) == T and len(swimbladder_x) == T and len(swimbladder_y) == T
        ):
            raise ValueError("All inputs must have the same number of time points (T).")

    @staticmethod
    def _validate_posture(head_x, head_y, head_yaw):
        T = len(head_x)
        if not (len(head_y) == T and head_yaw.shape[0] == T):
            raise ValueError("All inputs must have the same number of time points (T).")


class TailTrackingData(TrackingData):
    def __init__(self, tail_x, tail_y, tail_angle):
        self._tail_x = tail_x
        self._tail_y = tail_y
        self._tail_angle = tail_angle
        self.T = len(self._tail_x)

    @classmethod
    def from_keypoints(cls, *, tail_x, tail_y):
        cls._validate_keypoints(tail_x, tail_y)
        if tail_x.shape[1] != 11:
            tail_x, tail_y = interpolate_tail_keypoint(tail_x, tail_y, 10)
        tail_angle, head_yaw = compute_angles_from_keypoints(
            tail_x[:, 0] + 0.5, tail_y[:, 0], tail_x, tail_y
        )
        return cls(tail_x, tail_y, tail_angle)

    @classmethod
    def from_posture(cls, *, tail_angle):
        cls._validate_posture(tail_angle)
        if tail_angle.shape[1] != 10:
            tail_angle = interpolate_tail_angle(tail_angle, 10)
        T = tail_angle.shape[0]
        head_x, head_y, head_yaw = np.zeros(T), np.zeros(T), np.zeros(T)
        tail_x, tail_y = convert_tail_angle_to_keypoints(
            head_x,
            head_y,
            head_yaw,
            tail_angle,
            body_to_tail_mm=0.0,
            tail_to_tail_mm=0.32,
        )
        return cls(tail_x, tail_y, tail_angle)

    @property
    def tail_df(self):
        tail_df = pd.DataFrame(
            self._tail_angle, columns=[f"angle_{i}" for i in range(10)]
        )
        return tail_df

    @staticmethod
    def _validate_keypoints(tail_x, tail_y):
        if tail_x.shape[0] != tail_y.shape[0]:
            raise ValueError("All inputs must have the same number of time points (T).")
        N_keypoints = tail_x.shape[1]
        if N_keypoints < 4:
            raise ValueError(
                "At least 4 points from swim bladder to tail tips are required for tail tracking"
            )
        if tail_x.shape[1] != tail_y.shape[1]:
            raise ValueError(
                "tail_x and tail_y must have the same number of keypoints (N)."
            )

    @staticmethod
    def _validate_posture(tail_angle):
        N_keypoints = tail_angle.shape[1] + 1
        if N_keypoints < 4:
            raise ValueError(
                "At least 4 points from swim bladder to tail tips are required for full tracking"
            )
