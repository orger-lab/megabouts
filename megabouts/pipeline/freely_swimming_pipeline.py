import numpy as np


from ..config.preprocessing_config import (
    TailPreprocessingConfig,
    TrajPreprocessingConfig,
)
from ..preprocessing.traj_preprocessing import TrajPreprocessing
from ..preprocessing.tail_preprocessing import TailPreprocessing

from ..config.segmentation_config import TailSegmentationConfig, TrajSegmentationConfig
from ..segmentation.segmentation import Segmentation

from ..classification.classification import TailBouts, BoutClassifier

from ..utils.data_utils import create_hierarchical_df
from ..pipeline.base_pipeline import Pipeline


class EthogramHeadTracking:
    """Container for head tracking ethogram data.

    Parameters
    ----------
    segments : SegmentationResult
        Segmentation results
    bouts : TailBouts
        Classified bout data
    traj : TrajPreprocessingResult
        Preprocessed trajectory data
    """

    def __init__(self, segments, bouts, traj):
        self.segments = segments
        self.df = self.compute_df(bouts, traj)

    def compute_df(self, bouts, traj):
        head_x = traj.x
        head_y = traj.y
        head_angle = traj.yaw
        vigor = traj.vigor

        # is_swimming = segments.is_swimming
        bout_idx = self.compute_time_series(
            np.arange(len(bouts.category)), default_val=-1
        )
        bout_cat_ts = self.compute_time_series(bouts.category, default_val=-1)
        bout_sign_ts = self.compute_time_series(bouts.sign)

        data_info = [
            ("trajectory", "x", head_x),
            ("trajectory", "y", head_y),
            ("trajectory", "angle", head_angle),
            ("trajectory", "vigor", vigor),
            # ('bout','is_swimming',is_swimming),
            ("bout", "id", bout_idx),
            ("bout", "cat", bout_cat_ts),
            ("bout", "sign", bout_sign_ts),
        ]

        df = create_hierarchical_df(data_info)
        return df

    def compute_time_series(self, x, default_val=0):
        x_ts = np.full(self.segments.T, default_val)
        for i, (on_, off_) in enumerate(zip(self.segments.onset, self.segments.offset)):
            x_ts[on_:off_] = x[i]
        return x_ts


class EthogramFullTracking:
    """Container for full tracking ethogram data.

    Parameters
    ----------
    segments : SegmentationResult
        Segmentation results
    bouts : TailBouts
        Classified bout data
    tail : TailPreprocessingResult
        Preprocessed tail data
    traj : TrajPreprocessingResult
        Preprocessed trajectory data
    """

    def __init__(self, segments, bouts, tail, traj):
        self.segments = segments
        self.df = self.compute_df(bouts, tail, traj)

    def compute_df(self, bouts, tail, traj):
        tail_angle = tail.angle_smooth
        vigor = tail.vigor
        head_x = traj.x
        head_y = traj.y
        head_angle = traj.yaw

        # is_swimming = segments.is_swimming
        bout_idx = self.compute_time_series(
            np.arange(len(bouts.category)), default_val=-1
        )
        bout_cat_ts = self.compute_time_series(bouts.category, default_val=-1)
        bout_sign_ts = self.compute_time_series(bouts.sign)

        data_info = [
            ("tail_angle", "segment", tail_angle),
            ("tail_angle", "vigor", vigor),
            ("trajectory", "x", head_x),
            ("trajectory", "y", head_y),
            ("trajectory", "angle", head_angle),
            # ('bout','is_swimming',is_swimming),
            ("bout", "id", bout_idx),
            ("bout", "cat", bout_cat_ts),
            ("bout", "sign", bout_sign_ts),
        ]

        df = create_hierarchical_df(data_info)
        return df

    def compute_time_series(self, x, default_val=0):
        x_ts = np.full(self.segments.T, default_val)
        for i, (on_, off_) in enumerate(zip(self.segments.onset, self.segments.offset)):
            x_ts[on_:off_] = x[i]
        return x_ts


class HeadTrackingPipeline(Pipeline):
    """Pipeline for processing freely swimming fish with head tracking only.

    Parameters
    ----------
    tracking_cfg : TrackingConfig
        Configuration for tracking data
    exclude_CS : bool, optional
        Whether to exclude capture swim bouts, by default False

    Examples
    --------
    >>> import pandas as pd
    >>> from megabouts.tracking_data import TrackingConfig, HeadTrackingData, load_example_data
    >>> df, fps, mm_per_unit = load_example_data('fulltracking_posture')
    >>> head_x = df["head_x"].values * mm_per_unit
    >>> head_y = df["head_y"].values * mm_per_unit
    >>> head_yaw = df["head_angle"].values
    >>> tracking_data = HeadTrackingData.from_posture(
    ...     head_x=head_x, head_y=head_y, head_yaw=head_yaw
    ... )
    >>> tracking_cfg = TrackingConfig(fps=fps, tracking='head_tracking')
    >>> pipeline = HeadTrackingPipeline(tracking_cfg)
    >>> ethogram, bouts, segments, traj = pipeline.run(tracking_data)
    >>> isinstance(ethogram.df, pd.DataFrame)
    True
    """

    def __init__(self, tracking_cfg, exclude_CS=False):
        self.tracking_cfg = tracking_cfg
        # self.logger = logging.getLogger(__name__)
        # logging.basicConfig(level=logging.INFO)
        # self.logger.info("Initializing FullTrackingPipeline...")
        self.initialize_parameters_for_pipeline()
        self.exclude_CS = exclude_CS

    def initialize_parameters_for_pipeline(self):
        self.traj_preprocessing_cfg = TrajPreprocessingConfig(fps=self.tracking_cfg.fps)
        self.traj_segmentation_cfg = TrajSegmentationConfig(fps=self.tracking_cfg.fps)

    def preprocess_traj(self, traj_df):
        traj = TrajPreprocessing(self.traj_preprocessing_cfg).preprocess_traj_df(
            traj_df
        )
        return traj

    def segment_traj(self, traj_vigor):
        segmentation_function = Segmentation.from_config(self.traj_segmentation_cfg)
        segments = segmentation_function.segment(traj_vigor)
        return segments

    def classify_bouts(self, traj, segments):
        # Include Nan:
        x, y, yaw = traj.x_smooth, traj.y_smooth, traj.yaw_smooth
        x[traj.no_tracking], y[traj.no_tracking], yaw[traj.no_tracking] = (
            np.nan,
            np.nan,
            np.nan,
        )
        traj_array = segments.extract_traj_array(head_x=x, head_y=y, head_angle=yaw)
        classifier = BoutClassifier(
            self.tracking_cfg, self.traj_segmentation_cfg, exclude_CS=self.exclude_CS
        )
        classif_results = classifier.run_classification(traj_array=traj_array)
        segments.set_HB1(classif_results["first_half_beat"])

        traj_array = segments.extract_traj_array(
            head_x=x, head_y=y, head_angle=yaw, align_to_onset=False
        )

        bouts = TailBouts(
            segments=segments,
            classif_results=classif_results,
            tail_array=None,
            traj_array=traj_array,
        )

        return bouts

    # def compute_ethogram(self,tail_df,traj_df,segment_df,bouts_df):
    #    return segment_df

    def run(self, tracking_data):
        # self.logger.info("Running FullTrackingPipeline...")
        # self.logger.info("Preprocessing...")
        traj = self.preprocess_traj(tracking_data.traj_df)
        # self.logger.info("Segmentation...")
        segments = self.segment_traj(traj.vigor)
        # self.logger.info("Classification...")
        bouts = self.classify_bouts(traj, segments)

        ethogram = EthogramHeadTracking(segments, bouts, traj)

        return ethogram, bouts, segments, traj

    def __str__(self):
        lin1 = (
            f"Parameters are: traj_preprocessing_cfg: {self.traj_preprocessing_cfg}"
            + "\n"
        )
        lin2 = f"Parameters are: traj_segmentation_cfg: {self.traj_segmentation_cfg}"
        return lin1 + lin2

    def __repr__(self):
        return self.__str__()


class FullTrackingPipeline(Pipeline):
    """Pipeline for processing freely swimming fish with full tracking data.

    Parameters
    ----------
    tracking_cfg : TrackingConfig
        Configuration for tracking data
    exclude_CS : bool, optional
        Whether to exclude capture swim bouts, by default False

    Examples
    --------
    >>> import pandas as pd
    >>> from megabouts.tracking_data import TrackingConfig, FullTrackingData, load_example_data
    >>> df, fps, mm_per_unit = load_example_data('fulltracking_posture')
    >>> head_x = df["head_x"].values * mm_per_unit
    >>> head_y = df["head_y"].values * mm_per_unit
    >>> head_yaw = df["head_angle"].values
    >>> tail_angle = df.filter(like="tail_angle").values
    >>> tracking_data = FullTrackingData.from_posture(
    ...     head_x=head_x, head_y=head_y, head_yaw=head_yaw, tail_angle=tail_angle
    ... )
    >>> tracking_cfg = TrackingConfig(fps=fps, tracking='full_tracking')
    >>> pipeline = FullTrackingPipeline(tracking_cfg)
    >>> ethogram, bouts, segments, tail, traj = pipeline.run(tracking_data)
    >>> isinstance(ethogram.df, pd.DataFrame)
    True
    """

    def __init__(self, tracking_cfg, exclude_CS=False):
        self.tracking_cfg = tracking_cfg
        # self.logger = logging.getLogger(__name__)
        # logging.basicConfig(level=logging.INFO)
        # self.logger.info("Initializing FullTrackingPipeline...")
        self.initialize_parameters_for_pipeline()
        self.exclude_CS = exclude_CS

    def initialize_parameters_for_pipeline(self):
        self.tail_preprocessing_cfg = TailPreprocessingConfig(fps=self.tracking_cfg.fps)
        self.traj_preprocessing_cfg = TrajPreprocessingConfig(fps=self.tracking_cfg.fps)
        self.segmentation_cfg = TailSegmentationConfig(fps=self.tracking_cfg.fps)

    def preprocess_tail(self, tail_df):
        tail = TailPreprocessing(self.tail_preprocessing_cfg).preprocess_tail_df(
            tail_df
        )
        return tail

    def preprocess_traj(self, traj_df):
        traj = TrajPreprocessing(self.traj_preprocessing_cfg).preprocess_traj_df(
            traj_df
        )
        return traj

    def segment(self, vigor):
        segmentation_function = Segmentation.from_config(self.segmentation_cfg)
        segments = segmentation_function.segment(vigor)
        return segments

    def classify_bouts(self, tail, traj, segments):
        tail_array = segments.extract_tail_array(tail_angle=tail.angle_smooth)
        traj_array = segments.extract_traj_array(
            head_x=traj.x_smooth, head_y=traj.y_smooth, head_angle=traj.yaw_smooth
        )

        classifier = BoutClassifier(
            self.tracking_cfg, self.segmentation_cfg, exclude_CS=self.exclude_CS
        )
        classif_results = classifier.run_classification(
            tail_array=tail_array, traj_array=traj_array
        )
        segments.set_HB1(classif_results["first_half_beat"])

        tail_array = segments.extract_tail_array(
            tail_angle=tail.angle_smooth, align_to_onset=False
        )

        traj_array = segments.extract_traj_array(
            head_x=traj.x_smooth,
            head_y=traj.y_smooth,
            head_angle=traj.yaw_smooth,
            align_to_onset=False,
        )

        bouts = TailBouts(
            segments=segments,
            classif_results=classif_results,
            tail_array=tail_array,
            traj_array=traj_array,
        )

        return bouts

    def run(self, tracking_data):
        # self.logger.info("Running FullTrackingPipeline...")
        # self.logger.info("Preprocessing...")
        tail = self.preprocess_tail(tracking_data.tail_df)
        traj = self.preprocess_traj(tracking_data.traj_df)
        # self.logger.info("Segmentation...")
        if isinstance(self.segmentation_cfg, TailSegmentationConfig):
            segments = self.segment(tail.vigor)
        elif isinstance(self.segmentation_cfg, TrajSegmentationConfig):
            segments = self.segment(traj.vigor)
        else:
            raise ValueError(
                "segmentation_cfg should be an instance of TailSegmentationConfig or TrajSegmentationConfig"
            )

        # self.logger.info("Classification...")
        bouts = self.classify_bouts(tail, traj, segments)

        ethogram = EthogramFullTracking(segments, bouts, tail, traj)

        return ethogram, bouts, segments, tail, traj

    def __str__(self):
        lin1 = (
            f"Parameters are: tail_preprocessing_cfg: {self.tail_preprocessing_cfg}"
            + "\n"
        )
        lin2 = (
            f"Parameters are: traj_preprocessing_cfg: {self.traj_preprocessing_cfg}"
            + "\n"
        )
        lin3 = f"Parameters are: tail_segmentation_cfg: {self.segmentation_cfg}"
        return lin1 + lin2 + lin3

    def __repr__(self):
        return self.__str__()
