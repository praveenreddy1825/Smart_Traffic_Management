"""Tests for Module 2: VehicleDetector.

Uses bus.jpg, a real street photo bundled with the ultralytics package
(a bus with several people in front of it).
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import ultralytics

from app.detection.vehicle_detector import Detection, VehicleDetector, count_by_class
from app.detection.visualization import draw_detections
from app.utils.config import load_config

BUS_IMAGE = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def detector(config):
    # Loaded once for all tests in this file (loading the model takes a moment)
    return VehicleDetector.from_config(config)


@pytest.fixture(scope="module")
def bus_image():
    image = cv2.imread(str(BUS_IMAGE))
    assert image is not None, f"Test image missing: {BUS_IMAGE}"
    return image


def test_detects_the_bus_in_a_real_photo(detector, bus_image):
    detections = detector.detect(bus_image)
    assert any(d.class_name == "bus" for d in detections)


def test_only_configured_vehicle_classes_are_returned(detector, bus_image, config):
    # The photo contains people, but person detection is off by default
    allowed = set(config["detection"]["vehicle_classes"])
    detections = detector.detect(bus_image)
    assert all(d.class_name in allowed for d in detections)


def test_detection_fields_are_valid(detector, bus_image):
    height, width = bus_image.shape[:2]
    for detection in detector.detect(bus_image):
        x1, y1, x2, y2 = detection.bbox
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        assert detection.confidence >= detector.confidence_threshold
        assert 0.0 <= detection.confidence <= 1.0


def test_higher_threshold_never_gives_more_detections(detector, bus_image):
    original = detector.confidence_threshold
    try:
        detector.confidence_threshold = 0.10
        low = len(detector.detect(bus_image))
        detector.confidence_threshold = 0.80
        high = len(detector.detect(bus_image))
    finally:
        detector.confidence_threshold = original
    assert high <= low


def test_person_detection_can_be_enabled(config, bus_image):
    person_config = {**config, "detection": {**config["detection"], "include_person": True}}
    detections = VehicleDetector.from_config(person_config).detect(bus_image)
    assert any(d.class_name == "person" for d in detections)


def test_blank_image_gives_no_detections(detector):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(blank) == []


def test_unknown_class_name_raises(config):
    weights = Path(config["paths"]["models_dir"]) / config["detection"]["model_name"]
    with pytest.raises(ValueError, match="ambulance"):
        VehicleDetector(weights, class_names=["car", "ambulance"])


def test_center_and_per_frame_counts():
    detections = [
        Detection((0, 0, 10, 20), 2, "car", 0.9),
        Detection((10, 10, 30, 30), 2, "car", 0.8),
        Detection((0, 0, 4, 4), 7, "truck", 0.7),
    ]
    assert detections[0].center == (5, 10)
    assert count_by_class(detections) == {"car": 2, "truck": 1}


def test_tracking_keeps_same_id_across_frames(detector, bus_image):
    detector.reset_tracking()
    ids_per_frame = []
    for _ in range(5):  # the same (static) bus in 5 consecutive frames
        buses = [d for d in detector.track(bus_image) if d.class_name == "bus"]
        ids_per_frame.append(buses[0].track_id if buses else None)
    detector.reset_tracking()
    assert ids_per_frame[0] is not None
    assert len(set(ids_per_frame)) == 1  # one bus -> one ID, not five


def test_drawing_keeps_image_shape(detector, bus_image):
    image = bus_image.copy()
    draw_detections(image, detector.detect(image))
    assert image.shape == bus_image.shape
    assert not np.array_equal(image, bus_image)  # something was drawn
