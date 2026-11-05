"""Unit tests for Module 11A MultiApproachManager."""

import numpy as np
import pytest

from app.detection.vehicle_detector import Detection
from app.traffic.multi_approach_manager import APPROACH_DIRECTIONS, MultiApproachManager


class FakeDetector:
    """Mock detector to avoid loading neural network weights during fast unit tests."""
    def detect(self, image):
        return [
            Detection(bbox=(10, 10, 50, 50), class_id=2, class_name="car", confidence=0.9),
            Detection(bbox=(60, 60, 120, 120), class_id=5, class_name="bus", confidence=0.85),
        ]


def test_manager_subsystems_initialization():
    approaches_cfg = {
        "north": {"name": "North St", "video_source": None},
        "south": {"name": "South St", "video_source": None},
        "east": {"name": "East St", "video_source": None},
        "west": {"name": "West St", "video_source": None},
    }
    traffic_cfg = {
        "class_weights": {"car": 1.0, "bus": 2.5},
        "level_thresholds": {"medium": 5.0, "high": 10.0},
    }

    manager = MultiApproachManager(approaches_cfg, FakeDetector(), traffic_cfg)

    assert set(manager.counters.keys()) == set(APPROACH_DIRECTIONS)
    assert set(manager.analyzers.keys()) == set(APPROACH_DIRECTIONS)
    assert manager.manual_emergency_flags["north"] is False


def test_manual_emergency_triggering():
    approaches_cfg = {d: {"name": d.title(), "video_source": None} for d in APPROACH_DIRECTIONS}
    manager = MultiApproachManager(approaches_cfg, FakeDetector())

    manager.set_manual_emergency("north", True, "Ambulance")
    assert manager.manual_emergency_flags["north"] is True
    assert manager.manual_emergency_labels["north"] == "Ambulance"

    manager.set_manual_emergency("north", False)
    assert manager.manual_emergency_flags["north"] is False
    assert manager.manual_emergency_labels["north"] is None


def test_step_with_no_active_videos_returns_graceful_states():
    approaches_cfg = {d: {"name": d.title(), "video_source": None} for d in APPROACH_DIRECTIONS}
    manager = MultiApproachManager(approaches_cfg, FakeDetector())

    states = manager.step()
    assert len(states) == 4
    for d in APPROACH_DIRECTIONS:
        assert states[d].frame is None
        assert states[d].annotated_image is None
        assert states[d].has_visual_emergency is False

