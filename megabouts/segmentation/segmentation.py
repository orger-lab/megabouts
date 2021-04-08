from abc import ABC, abstractmethod

class BoutSegmenter(ABC):

    @abstractmethod
    def segment(self, raw_data):
        # process raw_data
        pass