"""
Configuration loader.

Reads config/config.yaml once and returns it as a plain Python dict.
Relative paths in the "paths" section are converted to absolute paths
based on the project root, so modules work no matter which folder
the program is started from.
"""

from pathlib import Path

import yaml

# app/utils/config.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

REQUIRED_SECTIONS = ["project", "paths", "logging", "device"]


def load_config(config_path=None):
    """Load the YAML config file and return it as a dict.

    Args:
        config_path: optional path to a different YAML file (used by tests).

    Raises:
        FileNotFoundError: if the config file does not exist.
        ValueError: if the file is empty or a required section is missing.
    """
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not config:
        raise ValueError(f"Config file is empty: {config_path}")

    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Config is missing required section(s): {missing}")

    # Turn every relative path into an absolute Path object
    config["paths"] = {
        name: resolve_path(value) for name, value in config["paths"].items()
    }
    return config


def resolve_path(path_value):
    """Return an absolute Path; relative paths are taken from the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_directories(config):
    """Create the data/models/logs folders from the config if they are missing."""
    for directory in config["paths"].values():
        Path(directory).mkdir(parents=True, exist_ok=True)
