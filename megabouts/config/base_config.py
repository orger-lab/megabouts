from dataclasses import dataclass, field


@dataclass
class BaseConfig:
    """Base configuration class.

    Parameters
    ----------
    fps : int
        Frames per second of the recording.
        Cannot be modified after initialization.

    Examples
    --------
    >>> config = BaseConfig(fps=30)
    >>> config.fps
    30
    >>> config.convert_ms_to_frames(1000)  # 1 second
    30
    """

    fps: int = field(init=True, repr=True)

    def __post_init__(self):
        """Initialize the fps attribute and ensure it cannot be modified later."""
        self._fps = self.fps

    @property
    def fps(self):  # noqa: RUF100, F811
        return self._fps

    @fps.setter
    def fps(self, value):
        if hasattr(self, "_fps"):
            raise AttributeError("Cannot modify fps after it has been set")
        self._fps = value

    def convert_ms_to_frames(self, milliseconds: float) -> int:
        """Convert milliseconds to an equivalent number of frames.

        Parameters
        ----------
        milliseconds : float
            Time in milliseconds

        Returns
        -------
        int
            Equivalent time in frames
        """
        return int(milliseconds / 1000 * self.fps)


class ConfigManager:
    """Manager for handling multiple configuration instances.

    Ensures consistency between different configuration components,
    particularly checking that fps values match across all configs.

    Parameters
    ----------
    *configs : BaseConfig
        Variable number of configuration instances

    Examples
    --------
    >>> from megabouts.config import TailPreprocessingConfig, TrajPreprocessingConfig
    >>> tail_cfg = TailPreprocessingConfig(fps=40)
    >>> traj_cfg = TrajPreprocessingConfig(fps=40)
    >>> cfg_manager = ConfigManager(tail_cfg, traj_cfg)
    Consistent FPS across all configurations: 40
    >>> cfg_manager.check_configs('tailpreprocessing')
    True
    """

    def __init__(self, *configs):
        """Initialize ConfigManager by categorizing configuration instances by their class type."""
        self.configs = {}
        for config in configs:
            config_class_name = config.__class__.__name__.lower()
            if config_class_name.endswith("config"):
                config_class_name = config_class_name[
                    :-6
                ]  # Remove 'config' from the name if present
            self.configs[config_class_name] = config

        self.check_fps_consistency()

    def check_fps_consistency(self):
        """Check if all configurations share the same fps value.

        Returns
        -------
        bool
            True if FPS is consistent across all configurations
        """
        fps_values = [
            config.fps for config in self.configs.values() if hasattr(config, "fps")
        ]
        if len(set(fps_values)) > 1:
            print(f"Inconsistent FPS across configurations: {fps_values}")
            return False
        print(
            f"Consistent FPS across all configurations: {fps_values[0] if fps_values else 'No FPS set'}"
        )
        return True

    def check_configs(self, *required_configs):
        """Check if all required configurations are provided.

        Parameters
        ----------
        *required_configs : str
            Names of required configuration components

        Returns
        -------
        bool
            True if all required configurations are present
        """
        adjusted_required_configs = [
            config.replace("_config", "") for config in required_configs
        ]
        missing_configs = [
            cfg for cfg in adjusted_required_configs if cfg not in self.configs
        ]
        if missing_configs:
            print(f"Missing configurations: {', '.join(missing_configs)}")
            return False
        return True
