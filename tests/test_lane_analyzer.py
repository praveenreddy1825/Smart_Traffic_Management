"""Tests for Module 5: LaneAnalyzer (no model needed)."""

import pytest

from app.detection.vehicle_detector import Detection
from app.lane.lane_analyzer import LaneAnalyzer, find_lane_profile, ground_point
from app.traffic.traffic_analyzer import HIGH, LOW, MEDIUM
from app.utils.config import load_config

FRAME_SIZE = (300, 100)
# Three vertical strips: x 0-100, 100-200, 200-300
THREE_LANES = [
    {"name": "Lane 1", "polygon": [[0.0, 0.0], [1 / 3, 0.0], [1 / 3, 1.0], [0.0, 1.0]]},
    {"name": "Lane 2", "polygon": [[1 / 3, 0.0], [2 / 3, 0.0], [2 / 3, 1.0], [1 / 3, 1.0]]},
    {"name": "Lane 3", "polygon": [[2 / 3, 0.0], [1.0, 0.0], [1.0, 1.0], [2 / 3, 1.0]]},
]
WEIGHTS = {"car": 1.0, "motorcycle": 0.5, "bus": 2.5, "truck": 2.5}


def vehicle(x1, y1, x2, y2, track_id=None, class_name="car"):
    return Detection((x1, y1, x2, y2), 2, class_name, 0.9, track_id=track_id)


def make_analyzer(**overrides):
    settings = dict(lanes=THREE_LANES, frame_size=FRAME_SIZE, class_weights=WEIGHTS,
                    density_window_seconds=1.0, medium_threshold=3.0, high_threshold=6.0,
                    min_frames_to_count=3)
    settings.update(overrides)
    return LaneAnalyzer(**settings)


def test_ground_point_is_bottom_center():
    assert ground_point(vehicle(10, 20, 30, 60)) == (20, 60)


def test_vehicle_assigned_to_lane_containing_its_ground_point():
    analyzer = make_analyzer()
    assert analyzer.assign_lane(vehicle(20, 10, 60, 50)) == "Lane 1"
    assert analyzer.assign_lane(vehicle(120, 10, 180, 50)) == "Lane 2"
    assert analyzer.assign_lane(vehicle(250, 10, 290, 50)) == "Lane 3"


def test_bottom_center_vs_center_for_a_slanted_lane():
    # Strongly slanted lane (perspective): x 0-100 at the top, x 200-300 at the bottom,
    # so at mid-height (y = 50) it spans x 100-200
    slanted = [{"name": "Lane A", "polygon": [[0.0, 0.0], [1 / 3, 0.0],
                                              [1.0, 1.0], [2 / 3, 1.0]]}]
    tall_bus = vehicle(160, 0, 260, 100, class_name="bus")
    # wheels (210, 100) are inside the lane ...
    assert make_analyzer(lanes=slanted).assign_lane(tall_bus) == "Lane A"
    # ... but the box centre (210, 50) floats outside it
    by_center = make_analyzer(lanes=slanted, assignment_point="center")
    assert by_center.assign_lane(tall_bus) is None


def test_vehicle_outside_all_lanes_is_ignored():
    analyzer = make_analyzer(lanes=THREE_LANES[:1])          # only Lane 1 defined
    reports = analyzer.update([vehicle(250, 10, 290, 50, track_id=1)], 0.0)
    assert reports[0].visible_now == 0


def test_per_lane_counts_and_levels():
    analyzer = make_analyzer()
    detections = (
        [vehicle(10 + i, 10, 30 + i, 50, track_id=i) for i in range(7)]          # 7 cars, lane 1
        + [vehicle(110 + i, 10, 130 + i, 50, track_id=10 + i) for i in range(4)]  # 4 cars, lane 2
        + [vehicle(210, 10, 230, 50, track_id=20)]                                # 1 car, lane 3
    )
    reports = analyzer.update(detections, 0.0)
    assert [r.visible_now for r in reports] == [7, 4, 1]
    assert [r.level for r in reports] == [HIGH, MEDIUM, LOW]
    assert reports[0].text() == "Lane 1 -> 7 vehicles -> HIGH"


def test_bus_weight_makes_lane_congested():
    analyzer = make_analyzer()
    buses = [vehicle(10, 10, 40, 50, 1, "bus"), vehicle(50, 10, 90, 50, 2, "bus"),
             vehicle(60, 10, 95, 50, 3, "bus")]
    report = analyzer.update(buses, 0.0)[0]
    assert report.load_now == pytest.approx(7.5)
    assert report.level == HIGH


def test_unique_vehicles_per_lane_need_min_frames():
    analyzer = make_analyzer(min_frames_to_count=3)
    for frame in range(5):
        reports = analyzer.update([vehicle(10, 10, 30, 50, track_id=1)], frame / 30)
    analyzer.update([vehicle(110, 10, 130, 50, track_id=2)], 6 / 30)          # 1 frame only
    reports = analyzer.update([], 7 / 30)
    assert reports[0].unique_vehicles == 1
    assert reports[1].unique_vehicles == 0


def test_lane_density_is_smoothed_over_time():
    analyzer = make_analyzer(density_window_seconds=1.0)
    cars = [vehicle(10 + i, 10, 30 + i, 50, track_id=i) for i in range(8)]
    for frame in range(30):
        analyzer.update(cars, frame / 30)
    report = analyzer.update([], 30 / 30)[0]      # one frame where nothing is detected
    assert report.level == HIGH


def test_profile_lookup_by_file_name():
    settings = {"profiles": {"clip.mp4": THREE_LANES}, "default_profile": None}
    assert find_lane_profile(settings, "data/videos/clip.mp4") == ("clip.mp4", THREE_LANES)
    assert find_lane_profile(settings, "other.mp4") == ("other.mp4", None)
    assert find_lane_profile(settings, 0)[0] == "webcam"
    settings["default_profile"] = "clip.mp4"
    assert find_lane_profile(settings, "other.mp4")[1] == THREE_LANES


def test_configured_profiles_are_valid():
    config = load_config()
    for profile, lanes in config["lanes"]["profiles"].items():
        assert len(lanes) >= 1, profile
        for lane in lanes:
            assert len(lane["polygon"]) >= 3
            assert all(0.0 <= value <= 1.0 for point in lane["polygon"] for value in point)


def test_no_profile_returns_none():
    config = load_config()
    assert LaneAnalyzer.from_config(config, "unknown_video.mp4", (960, 540)) is None
