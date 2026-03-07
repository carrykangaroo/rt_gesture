"""RT-Gesture core package."""

from .config import AppConfig, load_config, save_config

__version__ = "0.2.0"

__all__ = ["AppConfig", "load_config", "save_config", "__version__"]
