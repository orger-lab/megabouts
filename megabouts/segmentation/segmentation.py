from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import find_peaks
from ..utils.math_utils import find_onset_offset_numpy
from typing import Tuple
from ..config.segmentation_config import (
    SegmentationConfig,
    TailSegmentationConfig,
    TrajSegmentationConfig,
)


class SegmentationResult:
    """Container for segmentation results.

    Parameters
    ----------
    config : SegmentationConfig
        Configuration used for segmentation
    onset : np.ndarray
        Start frames of detected segments
    offset : np.ndarray
        End frames of detected segments
    T : int
        Total number of frames in recording
    """

    def __init__(
        self, config: SegmentationConfig, onset: np.ndarray, offset: np.ndarray, T: int
    ):
        self.config = config
        self.onset = onset.astype("int")
        self.offset = offset.astype("int")
        self.duration = self.offset - self.onset
        self.T = T
        self.HB1 = None

    def set_HB1(self, first_half_beat: np.ndarray):
        """Set the first half-beat frames for each segment.

        Parameters
        ----------
        first_half_beat : np.ndarray
            Frame indices of first half-beats, must match length of onset
        """
        if len(first_half_beat) != len(self.onset):
            raise ValueError(
                "Length of first_half_beat must be equal to the length of onset"
            )
        self.HB1 = (self.onset + first_half_beat).astype("int")
        # Make sure onset-offset include HB1
        self.onset[self.HB1 < self.onset] = self.HB1[self.HB1 < self.onset]

    def extract_tail_array(
        self, *, tail_angle: np.ndarray, align_to_onset: bool = True
    ) -> np.ndarray:
        """Extract tail angles for each detected segment.

        Parameters
        ----------
        tail_angle : np.ndarray
            Full tail angle array, shape (T, n_segments)
        align_to_onset : bool, optional
            If True, align to bout onset, else to HB1, by default True

        Returns
        -------
        np.ndarray
            Array of tail angles for each bout, shape (n_bouts, n_segments, bout_duration)
        """
        if align_to_onset:
            onset = self.onset
        else:
            onset = self.HB1 - self.config.first_half_beat_loc

        tail_array = np.zeros(
            (len(onset), tail_angle.shape[1], self.config.bout_duration)
        )
        for i, id_st in enumerate(onset):
            id_ed = id_st + self.config.bout_duration
            id_ed = min(id_ed, self.T - 1)
            dur = id_ed - id_st
            tail_array[i, :, :dur] = tail_angle[id_st:id_ed, :].T

        return tail_array

    def extract_traj_array(
        self,
        *,
        head_x: np.ndarray,
        head_y: np.ndarray,
        head_angle: np.ndarray,
        align_to_onset: bool = True,
        align: bool = True,
        idx_ref: int = 0,
    ) -> np.ndarray:
        """Extract trajectory data for each detected segment.

        Parameters
        ----------
        head_x, head_y : np.ndarray
            Head position coordinates
        head_angle : np.ndarray
            Head orientation angles
        align_to_onset : bool, optional
            If True, align to bout onset, else to HB1, by default True
        align : bool, optional
            Whether to align trajectories, by default True
        idx_ref : int, optional
            Reference frame for alignment, by default 0

        Returns
        -------
        np.ndarray
            Array of trajectory data for each bout, shape (n_bouts, 3, bout_duration)
            Channels are [x, y, angle]
        """
        if align_to_onset:
            onset = self.onset
        else:
            onset = self.HB1 - self.config.first_half_beat_loc

        x, y, head_angle = map(np.squeeze, [head_x, head_y, head_angle])

        traj_array = np.zeros((len(onset), 3, self.config.bout_duration))
        for i, id_st in enumerate(onset):
            id_ed = id_st + self.config.bout_duration
            id_ed = min(id_ed, self.T - 1)
            dur = id_ed - id_st
            sub_x, sub_y, sub_head_angle = (
                x[id_st:id_ed],
                y[id_st:id_ed],
                head_angle[id_st:id_ed],
            )
            traj_array[i, 0, :dur], traj_array[i, 1, :dur], traj_array[i, 2, :dur] = (
                sub_x,
                sub_y,
                sub_head_angle,
            )

        if align:
            traj_array = self.align_traj_array(traj_array=traj_array, idx_ref=idx_ref)

        return traj_array

    def align_traj_array(self, traj_array: np.ndarray, idx_ref: int) -> np.ndarray:
        """Align trajectory arrays to a reference point.

        Parameters
        ----------
        traj_array : np.ndarray
            Array of shape (N, 3, bout_duration) containing x, y, and heading
        idx_ref : int
            Reference index for alignment

        Returns
        -------
        np.ndarray
            Aligned trajectory array
        """
        return align_traj_array(traj_array, idx_ref, self.config.bout_duration)


class Segmentation(ABC):
    """Abstract base class for segmentation algorithms."""

    def __init__(self, config: SegmentationConfig):
        self.config = config

    @abstractmethod
    def segment(self, data: np.ndarray) -> SegmentationResult:
        """Perform segmentation on the provided data.

        Parameters
        ----------
        data : np.ndarray
            Data to segment

        Returns
        -------
        SegmentationResult
            Detected segments
        """
        pass

    @classmethod
    def from_config(cls, config: SegmentationConfig) -> "Segmentation":
        """Factory method to create appropriate segmentation instances.

        Parameters
        ----------
        config : SegmentationConfig
            Configuration for segmentation

        Returns
        -------
        Segmentation
            Instance of appropriate segmentation subclass
        """
        if isinstance(config, TailSegmentationConfig):
            return TailSegmentation(config)
        elif isinstance(config, TrajSegmentationConfig):
            return TrajSegmentation(config)
        else:
            raise ValueError(f"Unknown segmentation config: {config}")


class TailSegmentation(Segmentation):
    """Class for segmenting data based on tail movement."""

    def __init__(self, config: TailSegmentationConfig):
        super().__init__(config)

    def segment(self, tail_vigor: np.ndarray) -> SegmentationResult:
        """Segment data based on tail vigor.

        Parameters
        ----------
        tail_vigor : np.ndarray
            1D Array of tail vigor

        Returns
        -------
        SegmentationResult
            Detected segments
        """

        Thresh = self.config.threshold

        tail_active = tail_vigor > Thresh

        # Remove bouts of short duration
        onset, offset, duration = find_onset_offset_numpy(tail_active)
        onset = onset[duration > self.config.min_bout_duration]
        offset = offset[duration > self.config.min_bout_duration]
        segments = SegmentationResult(self.config, onset, offset, len(tail_vigor))
        return segments


class TrajSegmentation(Segmentation):
    """Class for segmenting data based on trajectory movement."""

    def __init__(self, config: TrajSegmentationConfig):
        super().__init__(config)

    def segment(self, kinematic_activity: np.ndarray) -> SegmentationResult:
        """Segment data based on kinematic activity.

        Parameters
        ----------
        kinematic_activity : np.ndarray
            Array of kinematic activity values

        Returns
        -------
        SegmentationResult
            Detected segments
        """

        peaks, _ = find_peaks(
            kinematic_activity,
            distance=self.config.bout_duration,
            prominence=self.config.peak_prominence,
        )
        inter_peak_min = TrajSegmentation.find_inter_peak_min(kinematic_activity, peaks)
        onset, offset = TrajSegmentation.find_onset_offset_around_peak(
            kinematic_activity, peaks, inter_peak_min, self.config.peak_percentage
        )

        segments = SegmentationResult(
            self.config, onset, offset, len(kinematic_activity)
        )

        return segments

    @staticmethod
    def find_inter_peak_min(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        """Find minima between peaks.

        Parameters
        ----------
        x : np.ndarray
            Input signal
        peaks : np.ndarray
            Indices of peaks

        Returns
        -------
        np.ndarray
            Indices of minima between peaks
        """
        peaks_list = [0] + peaks.tolist() + [len(x)]
        inter_peak_min = [
            p1 + np.argmin(x[p1:p2]) for p1, p2 in zip(peaks_list[:-1], peaks_list[1:])
        ]
        return np.array(inter_peak_min)

    @staticmethod
    def find_onset_offset_around_peak(
        x: np.ndarray,
        peaks: np.ndarray,
        inter_peak_min: np.ndarray,
        peak_percentage: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find onset and offset around each peak.

        Parameters
        ----------
        x : np.ndarray
            Input signal
        peaks : np.ndarray
            Indices of peaks
        inter_peak_min : np.ndarray
            Indices of minima between peaks
        peak_percentage : float
            Percentage of peak value to determine onset/offset

        Returns
        -------
        onset : np.ndarray
            Onset indices
        offset : np.ndarray
            Offset indices
        """
        onset = []
        offset = []

        for p_before, p, p_after in zip(inter_peak_min[:-1], peaks, inter_peak_min[1:]):
            # Find onset
            on_ = p
            while on_ > p_before and x[on_] >= 0.25 * x[p]:
                on_ -= 1

            # Find offset
            off_ = p
            while off_ < p_after and x[off_] >= peak_percentage * x[p]:
                off_ += 1

            onset.append(on_)
            offset.append(off_)

        return np.array(onset), np.array(offset)


def align_traj_array(
    traj_array: np.ndarray, idx_ref: int, bout_duration: int
) -> np.ndarray:
    """Align trajectory arrays to a reference point.

    Parameters
    ----------
    traj_array : np.ndarray
        Array of shape (N, 3, bout_duration) containing x, y, and heading
    idx_ref : int
        Reference index for alignment
    bout_duration : int
        Duration of bout

    Returns
    -------
    np.ndarray
        Aligned trajectory array

    Raises
    ------
    ValueError
        If idx_ref is negative or greater than bout_duration
        If traj_array does not have the expected shape

    Examples
    --------
    >>> N, duration = 10, 100  # 10 bouts, 100 frames each
    >>> traj = np.zeros((N, 3, duration))  # x, y, heading
    >>> traj[:, 0, :] = np.linspace(0, 1, duration)  # x increases linearly
    >>> aligned = align_traj_array(traj, idx_ref=0, bout_duration=duration)
    >>> np.allclose(aligned[:, 0, 0], 0)  # all trajectories start at x=0
    True
    """
    if (
        not isinstance(traj_array, np.ndarray)
        or len(traj_array.shape) != 3
        or traj_array.shape[1] != 3
    ):
        raise ValueError(
            f"traj_array must be a numpy array of shape (N, 3, {bout_duration}), got shape {traj_array.shape}"
        )

    if idx_ref < 0 or idx_ref >= bout_duration:
        raise ValueError(
            f"idx_ref must be between 0 and {bout_duration-1}, got {idx_ref}"
        )

    if traj_array.shape[2] != bout_duration:
        raise ValueError(
            f"traj_array must have shape (N, 3, {bout_duration}), got shape {traj_array.shape}"
        )

    traj_array_aligned = np.zeros_like(traj_array)
    N = traj_array.shape[0]
    for i in range(N):
        sub_x, sub_y, sub_head_angle = (
            traj_array[i, 0, :],
            traj_array[i, 1, :],
            traj_array[i, 2, :],
        )
        Pos = np.zeros((2, bout_duration))
        Pos[0, :] = sub_x - sub_x[idx_ref]
        Pos[1, :] = sub_y - sub_y[idx_ref]
        theta = -sub_head_angle[idx_ref]
        head_angle_rotated = sub_head_angle - sub_head_angle[idx_ref]
        RotMat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        PosRot = np.dot(RotMat, Pos)
        sub_x, sub_y, sub_head_angle = (
            PosRot[0, :],
            PosRot[1, :],
            head_angle_rotated,
        )
        (
            traj_array_aligned[i, 0, :],
            traj_array_aligned[i, 1, :],
            traj_array_aligned[i, 2, :],
        ) = sub_x, sub_y, sub_head_angle

    return traj_array_aligned
