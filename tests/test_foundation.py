"""Tests for Module 0: configuration loading and logging."""

import logging

import pytest

from app.utils.config import PROJECT_ROOT, load_config
from app.utils.logger import get_logger, setup_logging


def test_default_config_loads_with_required_sections():
    config = load_config()
    for section in ["project", "paths", "logging", "device"]:
        assert section in config


def test_relative_paths_become_absolute_inside_project():
    config = load_config()
    for path in config["paths"].values():
        assert path.is_absolute()
        assert PROJECT_ROOT in path.parents


def test_missing_config_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_config_missing_section_raises(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("project:\n  name: test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required section"):
        load_config(bad_config)


def test_logging_writes_to_file(tmp_path):
    config = {
        "paths": {"logs_dir": tmp_path},
        "logging": {"level": "INFO", "log_to_file": True, "log_file": "test.log"},
    }
    setup_logging(config)
    get_logger("test_module").info("hello from test")

    # Flush and detach handlers so the temp file can be read and deleted on Windows
    for handler in list(logging.getLogger().handlers):
        handler.flush()
        handler.close()
        logging.getLogger().removeHandler(handler)

    log_text = (tmp_path / "test.log").read_text(encoding="utf-8")
    assert "hello from test" in log_text
    assert "test_module" in log_text
