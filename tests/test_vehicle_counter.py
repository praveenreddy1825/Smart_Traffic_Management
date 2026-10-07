"""Tests for Module 3: unique counting logic (no model needed - hand-made detections)."""

from app.detection.vehicle_detector import Detection
from app.tracking.vehicle_counter import VehicleCounter, line_to_pixels


def box_at(center_x, center_y, track_id, class_name="car", size=20):
    half = size / 2
    return Detection((center_x - half, center_y - half, center_x + half, center_y + half),
                     2, class_name, 0.9, track_id=track_id)


def test_same_id_over_four_frames_counts_once():
    # The example from the project brief: Car ID 1 in frames 1-4 -> Count = 1
    counter = VehicleCounter(min_frames_to_count=1)
    for frame in range(4):
        result = counter.update([box_at(100 + frame, 100, track_id=1)], frame)
    assert result.total_unique == 1
    assert result.unique_by_class == {"car": 1}


def test_two_different_vehicles_count_two():
    counter = VehicleCounter(min_frames_to_count=3)
    for frame in range(5):
        result = counter.update([box_at(50, 50, 1), box_at(200, 200, 2, "bus")], frame)
    assert result.total_unique == 2
    assert result.unique_by_class == {"car": 1, "bus": 1}
    assert result.visible_now == 2


def test_short_flicker_is_not_counted():
    counter = VehicleCounter(min_frames_to_count=3)
    counter.update([box_at(50, 50, 1)], 0)
    counter.update([box_at(51, 50, 1)], 1)       # only 2 frames
    for frame in range(2, 6):
        result = counter.update([], frame)
    assert result.total_unique == 0


def test_class_is_majority_vote():
    counter = VehicleCounter(min_frames_to_count=1)
    labels = ["car", "truck", "car", "car", "truck"]
    for frame, label in enumerate(labels):
        result = counter.update([box_at(100, 100, 7, label)], frame)
    assert result.unique_by_class == {"car": 1}


def test_untracked_detections_are_ignored():
    counter = VehicleCounter(min_frames_to_count=1)
    untracked = Detection((0, 0, 10, 10), 2, "car", 0.9, track_id=None)
    assert counter.update([untracked], 0).total_unique == 0


def test_line_crossing_counted_once_with_direction():
    line = ((0, 100), (300, 100))                 # horizontal line at y = 100
    counter = VehicleCounter(min_frames_to_count=1, counting_line=line)
    # Vehicle moves down across the line, then wobbles back up and down again
    for frame, y in enumerate([80, 95, 105, 95, 110, 130]):
        result = counter.update([box_at(150, y, 1)], frame)
    assert result.line_crossings == 1
    assert result.line_crossings_by_direction == {"down": 1}


def test_line_crossing_up_direction_and_no_crossing():
    line = ((0, 100), (300, 100))
    counter = VehicleCounter(min_frames_to_count=1, counting_line=line)
    for frame, y in enumerate([150, 120, 90, 60]):       # vehicle 1 moves UP across
        counter.update([box_at(100, y, 1), box_at(200, 300 - frame, 2)], frame)  # 2 never crosses
    result = counter.result()
    assert result.line_crossings == 1
    assert result.line_crossings_by_direction == {"up": 1}


def test_crossing_outside_line_segment_is_ignored():
    line = ((0, 100), (100, 100))                 # short line, x = 0..100 only
    counter = VehicleCounter(min_frames_to_count=1, counting_line=line)
    for frame, y in enumerate([80, 120]):
        result = counter.update([box_at(250, y, 1)], frame)   # passes at x = 250
    assert result.line_crossings == 0


def test_counts_survive_forgetting_old_tracks():
    counter = VehicleCounter(min_frames_to_count=1, forget_after_frames=5)
    for frame in range(3):
        counter.update([box_at(10, 10, 1, "truck")], frame)
    for frame in range(3, 20):
        result = counter.update([], frame)
    assert 1 not in counter.tracks                 # details forgotten
    assert result.unique_by_class == {"truck": 1}  # count kept


def test_line_fractions_to_pixels():
    assert line_to_pixels([[0.0, 0.5], [1.0, 0.5]], (960, 540)) == ((0.0, 270.0), (960.0, 270.0))
    assert line_to_pixels(None, (960, 540)) is None
