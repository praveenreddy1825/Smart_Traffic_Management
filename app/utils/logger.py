"""
Logging setup.

Call setup_logging(config) once at program start. After that, every module
gets its own named logger with get_logger(__name__), so each log line shows
which module wrote it:

    2026-10-01 10:15:02 | INFO | app.video.video_input | Opened video ...
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(config):
    """Configure console (and optionally file) logging from the config dict."""
    log_settings = config["logging"]
    level = getattr(logging, str(log_settings.get("level", "INFO")).upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove handlers from a previous call (e.g. Streamlit re-runs the script)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_settings.get("log_to_file", True):
        logs_dir = Path(config["paths"]["logs_dir"])
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            logs_dir / log_settings.get("log_file", "sentinel.log"),
            maxBytes=int(log_settings.get("max_file_size_mb", 5)) * 1024 * 1024,
            backupCount=int(log_settings.get("backup_count", 3)),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name):
    """Return a named logger for a module (use get_logger(__name__))."""
    return logging.getLogger(name)
