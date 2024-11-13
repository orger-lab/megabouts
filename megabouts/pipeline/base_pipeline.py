from abc import ABC, abstractmethod


class Pipeline(ABC):
    @abstractmethod
    def initialize_parameters_for_pipeline(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    @abstractmethod
    def run(self, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses")
