from dataclasses import dataclass

from .base_config import BaseConfig


@dataclass
class SegmentationConfig(BaseConfig):
    """Configuration for generic preprocessing.

    Attributes:
        bout_duration_ms (float): Duration of the bout in milliseconds.
        first_half_beat_loc_ms (float): Location of first half beat within a bout in milliseconds.
    """

    bout_duration_ms: float = 200
    first_half_beat_loc_ms: float = 36

    @property
    def bout_duration(self):
        """Convert the minimum segment size from milliseconds to frames."""
        return self.convert_ms_to_frames(self.bout_duration_ms)

    @property
    def first_half_beat_loc(self):
        """Convert the first half beat location from milliseconds to frames."""
        return self.convert_ms_to_frames(self.first_half_beat_loc_ms)


@dataclass
class TailSegmentationConfig(SegmentationConfig):
    """Configuration for tail segmentation.

    Attributes:
        threshold (float):  Threshold for tail speed
        min_bout_duration_ms (float): Minimum bout duration in milliseconds.
    """

    min_bout_duration_ms: float = 85
    threshold: float = 100.0

    @property
    def min_bout_duration(self):
        """Convert the minimum segment size from milliseconds to frames."""
        return self.convert_ms_to_frames(self.min_bout_duration_ms)


@dataclass
class TrajSegmentationConfig(SegmentationConfig):
    """Configuration for traj segmentation.

    Attributes:
        peak_prominence (float): prominence for peak finding algorithm
        margin_before_peak_ms (float): Margin from peak to onset in milliseconds.
    """

    peak_prominence: float = 0.4
    peak_percentage: float = 0.2
    # margin_before_peak_ms: float = 28

    # @property
    # def margin_before_peak(self):
    #    """Convert the minimum segment size from milliseconds to frames."""
    #    return self.convert_ms_to_frames(self.margin_before_peak_ms)
