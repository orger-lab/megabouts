from abc import ABC, abstractmethod


class Pipeline(ABC):
    """Abstract base class for analysis pipelines."""

    @abstractmethod
    def initialize_parameters_for_pipeline(self):
        """Initialize pipeline parameters."""
        pass

    @abstractmethod
    def run(self, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses")
