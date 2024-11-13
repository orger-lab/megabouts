from ..config.preprocessing_config import TailPreprocessingConfig
from ..preprocessing.tail_preprocessing import TailPreprocessing

from ..config.segmentation_config import TailSegmentationConfig
from ..segmentation.segmentation import Segmentation


from ..config.sparse_coding_config import SparseCodingConfig
from ..sparse_coding.sparse_coding import SparseCoding

from ..pipeline.base_pipeline import Pipeline


class HeadRestrainedPipeline(Pipeline):
    def __init__(self, tracking_cfg):
        self.tracking_cfg = tracking_cfg
        self.initialize_parameters_for_pipeline()

    def initialize_parameters_for_pipeline(self):
        self.tail_preprocessing_cfg = TailPreprocessingConfig(fps=self.tracking_cfg.fps)
        self.tail_segmentation_cfg = TailSegmentationConfig(fps=self.tracking_cfg.fps)
        self.sparse_coding_cfg = SparseCodingConfig(fps=self.tracking_cfg.fps)

    def preprocess_tail(self, tail_df):
        tail = TailPreprocessing(self.tail_preprocessing_cfg).preprocess_tail_df(
            tail_df
        )
        return tail

    def segment_tail(self, tail_vigor):
        segmentation_function = Segmentation.from_config(self.tail_segmentation_cfg)
        segments = segmentation_function.segment(tail_vigor)
        return segments

    def compute_sparse_coding(self, tail_angle):
        sparse_coding = SparseCoding(self.sparse_coding_cfg)
        sparse_coding_result = sparse_coding.sparse_code_tail_angle(tail_angle)
        return sparse_coding_result

    def run(self, tracking_data):
        tail = self.preprocess_tail(tracking_data.tail_df)
        segments = self.segment_tail(tail.vigor)
        sparse_coding_result = self.compute_sparse_coding(tail.angle_smooth)
        return sparse_coding_result, segments, tail
