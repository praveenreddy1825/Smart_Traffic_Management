"""Tests for Module 4: TrafficAnalyzer (no model needed)."""

import pytest

from app.tracking.vehicle_counter import CountingResult
from app.traffic.traffic_analyzer import (
    HIGH, LOW, MEDIUM, TrafficAnalyzer, classify_level, weighted_load)

WEIGHTS = {"car": 1.0, "motorcycle": 0.5, "bus": 2.5, "truck": 2.5}


def counts(visible_by_class, total_unique=0, crossings=0, unique_by_class=None):
    return CountingResult(
        total_unique=total_unique,
        unique_by_class=unique_by_class or {},
        line_crossings=crossings,
        line_crossings_by_class={},
        line_crossings_by_direction={},
        visible_now=sum(visible_by_class.values()),
        visible_by_class=visible_by_class,
    )


def make_analyzer(**overrides):
    settings = dict(class_weights=WEIGHTS, density_window_seconds=3.0,
                    medium_threshold=8.0, high_threshold=15.0, flow_window_seconds=60.0)
    settings.update(overrides)
    return TrafficAnalyzer(**settings)


def test_weighted_load_uses_class_weights():
    # 4 x 1.0 + 2 x 0.5 + 1 x 2.5 = 7.5
    assert weighted_load({"car": 4, "motorcycle": 2, "bus": 1}, WEIGHTS) == pytest.approx(7.5)
    assert weighted_load({"tractor": 3}, WEIGHTS) == 3  # unknown class counts as 1.0


@pytest.mark.parametrize("density, expected", [
    (0, LOW), (7.99, LOW), (8.0, MEDIUM), (14.99, MEDIUM), (15.0, HIGH), (40, HIGH)])
def test_level_boundaries(density, expected):
    assert classify_level(density, 8.0, 15.0) == expected


def test_steady_heavy_traffic_is_high():
    analyzer = make_analyzer()
    for frame in range(90):  # 3 s at 30 FPS, 12 cars + 2 buses = load 17
        report = analyzer.update(counts({"car": 12, "bus": 2}), frame / 30)
    assert report.load_now == pytest.approx(17)
    assert report.level == HIGH


def test_single_empty_frame_does_not_drop_the_level():
    analyzer = make_analyzer()
    for frame in range(90):
        analyzer.update(counts({"car": 17}), frame / 30)
    report = analyzer.update(counts({}), 90 / 30)   # detector misses everything once
    assert report.load_now == 0
    assert report.level == HIGH                       # smoothed density still high


def test_density_window_uses_video_time():
    analyzer = make_analyzer(density_window_seconds=1.0)
    for frame in range(30):                          # 1 s of heavy traffic
        analyzer.update(counts({"car": 20}), frame / 30)
    for frame in range(30, 90):                      # then 2 s of empty road
        report = analyzer.update(counts({}), frame / 30)
    assert report.density == 0                       # old heavy frames left the window
    assert report.level == LOW


def test_flow_per_minute_from_line_crossings():
    analyzer = make_analyzer()
    analyzer.update(counts({}, crossings=0), 0.0)
    report = analyzer.update(counts({}, crossings=5), 30.0)   # 5 vehicles in 30 s
    assert report.flow_per_minute == pytest.approx(10.0)


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        make_analyzer(medium_threshold=20, high_threshold=10)


def test_summary_text_matches_brief_format():
    analyzer = make_analyzer()
    report = analyzer.update(
        counts({"car": 20}, total_unique=24,
               unique_by_class={"car": 17, "motorcycle": 4, "bus": 2, "truck": 1}), 0.0)
    text = report.summary_text()
    for expected in ["Total vehicles: 24", "Cars: 17", "Motorcycles: 4",
                     "Buses: 2", "Trucks: 1", "Traffic Level: HIGH"]:
        assert expected in text
