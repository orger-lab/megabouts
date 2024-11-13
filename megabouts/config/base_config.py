from dataclasses import dataclass, field


@dataclass
class BaseConfig:
    """Base configuration class.

    Attributes:
        fps (int): frames per second.
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
        Args:
            milliseconds (float): Time in milliseconds.
        Returns:
            int: Equivalent time in frames.
        Example:
            >>> from megabouts.config.component_configs import TailPreprocessingConfig
            >>> config = TailPreprocessingConfig(fps=30)
            >>> config.convert_ms_to_frames(1000)
            30
        """
        return int(milliseconds / 1000 * self.fps)


class ConfigManager:
    """Manager for handling multiple configuration instances.
    Args:
        *configs: Variable length configuration list.
    Attributes:
        configs (dict): Dictionary storing configuration instances categorized by their class type.
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
        Returns:
            bool: True if FPS is consistent across all configurations, False otherwise.
        Example:
            >>> from megabouts.config.base_config import ConfigManager
            >>> from megabouts.config.component_configs import TailPreprocessingConfig
            >>> from megabouts.config.component_configs import TrajPreprocessingConfig
            >>> fps = 40
            >>> tail_preprocessing_cfg= TailPreprocessingConfig(fps=fps)
            >>> traj_preprocessing_cfg = TrajPreprocessingConfig(fps=fps)
            >>> cfg_manager = ConfigManager(tail_preprocessing_cfg, traj_preprocessing_cfg)
            Consistent FPS across all configurations: 40
            >>> cfg_manager.check_fps_consistency()
            Consistent FPS across all configurations: 40
            True
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

        Args:
            *required_configs: Variable length list of required configuration names.

        Returns:
            bool: True if all required configurations are present, False otherwise.

        Example:
            >>> from megabouts.config.base_config import ConfigManager
            >>> from megabouts.config.component_configs import TailPreprocessingConfig
            >>> tail_preprocessing_cfg = TailPreprocessingConfig(fps=30)
            >>> cfg_manager = ConfigManager(tail_preprocessing_cfg)
            Consistent FPS across all configurations: 30
            >>> cfg_manager.check_configs('tailpreprocessing')
            True
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
