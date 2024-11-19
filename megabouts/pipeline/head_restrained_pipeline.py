from ..config.preprocessing_config import TailPreprocessingConfig
from ..preprocessing.tail_preprocessing import TailPreprocessing

from ..config.segmentation_config import TailSegmentationConfig
from ..segmentation.segmentation import Segmentation


from ..config.sparse_coding_config import SparseCodingConfig
from ..sparse_coding.sparse_coding import SparseCoding

from ..pipeline.base_pipeline import Pipeline


class HeadRestrainedPipeline(Pipeline):
    """Pipeline for processing head-restrained fish data.

    Parameters
    ----------
    tracking_cfg : TrackingConfig
        Configuration for tracking data

    Examples
    --------
    >>> import pandas as pd
    >>> from megabouts.tracking_data import TrackingConfig, TailTrackingData, load_example_data
    >>> df, fps, mm_per_unit = load_example_data('HR_DLC')
    >>> tail_x = df["DLC_resnet50_Zebrafish"].loc[:, [(f"tail{i}", "x") for i in range(11)]].values * mm_per_unit
    >>> tail_y = df["DLC_resnet50_Zebrafish"].loc[:, [(f"tail{i}", "y") for i in range(11)]].values * mm_per_unit
    >>> tracking_data = TailTrackingData.from_keypoints(tail_x=tail_x, tail_y=tail_y)
    >>> tracking_cfg = TrackingConfig(fps=fps, tracking='tail_tracking')
    >>> pipeline = HeadRestrainedPipeline(tracking_cfg)
    >>> sparse_coding_result, segments, tail = pipeline.run(tracking_data)
    >>> isinstance(sparse_coding_result.df, pd.DataFrame)
    True
    """

    def __init__(self, tracking_cfg):
        self.tracking_cfg = tracking_cfg
        self.initialize_parameters_for_pipeline()

    def initialize_parameters_for_pipeline(self):
        self.tail_preprocessing_cfg = TailPreprocessingConfig(
            fps=self.tracking_cfg.fps,
            baseline_method="whittaker",
            baseline_params={"lmbda": 1e5, "half_window": 200},
        )
        self.tail_segmentation_cfg = TailSegmentationConfig(fps=self.tracking_cfg.fps)
        self.sparse_coding_cfg = SparseCodingConfig(fps=self.tracking_cfg.fps)

    def preprocess_tail(self, tail_df):
        """Preprocess tail angle data.

        Parameters
        ----------
        tail_df : pd.DataFrame
            DataFrame containing tail angle data

        Returns
        -------
        TailPreprocessingResult
            Preprocessed tail data
        """
        tail = TailPreprocessing(self.tail_preprocessing_cfg).preprocess_tail_df(
            tail_df
        )
        return tail

    def segment_tail(self, tail_vigor):
        """Segment tail movement into bouts.

        Parameters
        ----------
        tail_vigor : np.ndarray
            Tail vigor signal

        Returns
        -------
        SegmentationResult
            Detected segments
        """
        segmentation_function = Segmentation.from_config(self.tail_segmentation_cfg)
        segments = segmentation_function.segment(tail_vigor)
        return segments

    def compute_sparse_coding(self, tail_angle):
        """Compute sparse coding of tail angles.

        Parameters
        ----------
        tail_angle : np.ndarray
            Tail angle data

        Returns
        -------
        SparseCodingResult
            Sparse coding results
        """
        sparse_coding = SparseCoding(self.sparse_coding_cfg)
        sparse_coding_result = sparse_coding.sparse_code_tail_angle(tail_angle)
        return sparse_coding_result

    def run(self, tracking_data):
        tail = self.preprocess_tail(tracking_data.tail_df)
        segments = self.segment_tail(tail.vigor)
        sparse_coding_result = self.compute_sparse_coding(tail.angle_smooth)
        return sparse_coding_result, segments, tail
