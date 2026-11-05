"""
Tests for Module 15: Edge Deployment Profiling and Offline Readiness Verification.
"""

from pathlib import Path
import pytest

from scripts.profile_edge_deployment import check_offline_weights, get_process_memory_mb
from app.utils.config import load_config, resolve_path


def test_offline_weights_presence():
    """Verify that YOLO and Hugging Face AST weights are cached locally."""
    config = load_config()
    weight_info = check_offline_weights(config)

    assert "yolo_model" in weight_info
    assert "ast_model" in weight_info
    assert weight_info["yolo_model"]["cached"] is True
    assert weight_info["yolo_model"]["size_mb"] > 0
    assert weight_info["ast_model"]["cached"] is True


def test_process_memory_reporting():
    """Verify that RSS process memory is returned as a valid float."""
    rss_mb = get_process_memory_mb()
    assert isinstance(rss_mb, float)
    assert rss_mb >= 0.0

