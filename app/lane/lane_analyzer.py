"""
Module 5 - Lane / Region Analysis.

Lanes are drawn BY HAND as polygons in config.yaml (no automatic lane
detection). Every tracked vehicle is placed in the lane that contains its
"ground point", and each lane gets its own count, density and level:

    lane_analyzer = LaneAnalyzer.from_config(config, video_source, frame_size)
    reports = lane_analyzer.update(tracked_detections, frame.timestamp_sec)
    for lane in reports:
        print(lane.text())            # "Lane 1 -> 5 vehicles -> MEDIUM"

Ground point = bottom-centre of the box (where the wheels touch the road).
With a camera looking down at an angle, the CENTRE of a tall bus can float over
the neighbouring lane, but its bottom edge stays in its own lane.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.traffic.traffic_analyzer import classify_level, weighted_load
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LaneReport:
    name: str
    visible_now: int          # vehicles in this lane in the current frame
    visible_by_class: dict
    load_now: float           # size-weighted vehicles now (same weights as Module 4)
    density: float            # load averaged over the density window
    level: str                # LOW / MEDIUM / HIGH (lane thresholds)
    unique_vehicles: int      # different vehicles that have been in this lane

    def text(self):
        return f"{self.name} -> {self.visible_now} vehicles -> {self.level}"


def ground_point(detection):
    """Bottom-centre of the bounding box."""
    x1, _, x2, y2 = detection.bbox
    return ((x1 + x2) / 2, y2)


def polygon_to_pixels(polygon_fractions, frame_size):
    """[[fx, fy], ...] with 0-1 fractions -> NumPy array of pixel points for OpenCV."""
    width, height = frame_size
    return np.array([[fx * width, fy * height] for fx, fy in polygon_fractions],
                    dtype=np.float32)


def find_lane_profile(lane_settings, video_source):
    """Pick the lane polygons for this video (profiles are keyed by video file name)."""
    profiles = lane_settings.get("profiles") or {}
    key = "webcam" if str(video_source).strip().isdigit() else Path(str(video_source)).name
    if key in profiles:
        return key, profiles[key]
    default_key = lane_settings.get("default_profile")
    if default_key and default_key in profiles:
        logger.warning("No lane profile for '%s' - using default profile '%s'", key, default_key)
        return default_key, profiles[default_key]
    return key, None


class LaneAnalyzer:
    def __init__(self, lanes, frame_size, class_weights=None, density_window_seconds=3.0,
                 medium_threshold=3.0, high_threshold=6.0, min_frames_to_count=10,
                 assignment_point="bottom_center"):
        """
        lanes: list of {"name": str, "polygon": [[fx, fy], ...]} with 0-1 fractions.
        frame_size: (width, height) of the processed frames.
        """
        if not lanes:
            raise ValueError("At least one lane must be defined")
        if medium_threshold >= high_threshold:
            raise ValueError("lane thresholds: 'medium' must be smaller than 'high'")
        self.lane_names = [lane["name"] for lane in lanes]
        self.polygons = {lane["name"]: polygon_to_pixels(lane["polygon"], frame_size)
                         for lane in lanes}
        self.class_weights = dict(class_weights or {})
        self.density_window_seconds = density_window_seconds
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.min_frames_to_count = min_frames_to_count
        self.assignment_point = assignment_point
        self.reset()

    @classmethod
    def from_config(cls, config, video_source, frame_size):
        """Returns None if no lane profile exists for this video (lane analysis is skipped)."""
        settings = config["lanes"]
        profile_name, lanes = find_lane_profile(settings, video_source)
        if not lanes:
            logger.warning("No lane regions defined for '%s' - lane analysis disabled. "
                           "Create them with scripts/define_lanes.py", profile_name)
            return None
        logger.info("Using lane profile '%s' with %d lanes", profile_name, len(lanes))
        thresholds = settings["level_thresholds"]
        return cls(
            lanes=lanes,
            frame_size=frame_size,
            class_weights=config.get("traffic", {}).get("class_weights"),
            density_window_seconds=settings.get("density_window_seconds", 3.0),
            medium_threshold=thresholds["medium"],
            high_threshold=thresholds["high"],
            min_frames_to_count=config.get("tracking", {}).get("min_frames_to_count", 10),
            assignment_point=settings.get("assignment_point", "bottom_center"),
        )

    def reset(self):
        self._recent_loads = {name: deque() for name in self.lane_names}
        # frames each track has spent in each lane -> used for per-lane unique counts
        self._frames_in_lane = {name: defaultdict(int) for name in self.lane_names}

    def assign_lane(self, detection):
        """Name of the lane containing the vehicle's assignment point, or None."""
        if self.assignment_point == "center":
            point = detection.center
        else:
            point = ground_point(detection)
        point = (float(point[0]), float(point[1]))
        for name in self.lane_names:
            # >= 0 means inside or exactly on the border
            if cv2.pointPolygonTest(self.polygons[name], point, False) >= 0:
                return name
        return None

    def update(self, detections, timestamp_sec):
        """Assign this frame's tracked detections to lanes; returns one LaneReport per lane."""
        visible = {name: defaultdict(int) for name in self.lane_names}
        for detection in detections:
            lane = self.assign_lane(detection)
            if lane is None:
                continue  # vehicle outside all lanes (e.g. pavement, far background)
            visible[lane][detection.class_name] += 1
            if detection.track_id is not None:
                self._frames_in_lane[lane][detection.track_id] += 1

        reports = []
        for name in self.lane_names:
            visible_by_class = dict(visible[name])
            load_now = weighted_load(visible_by_class, self.class_weights)
            density = self._rolling_average(name, timestamp_sec, load_now)
            unique = sum(1 for frames in self._frames_in_lane[name].values()
                         if frames >= self.min_frames_to_count)
            reports.append(LaneReport(
                name=name,
                visible_now=sum(visible_by_class.values()),
                visible_by_class=visible_by_class,
                load_now=load_now,
                density=density,
                level=classify_level(density, self.medium_threshold, self.high_threshold),
                unique_vehicles=unique,
            ))
        return reports

    def _rolling_average(self, lane_name, timestamp_sec, load_now):
        """Average lane load over the last density_window_seconds of video time."""
        window = self._recent_loads[lane_name]
        window.append((timestamp_sec, load_now))
        while timestamp_sec - window[0][0] > self.density_window_seconds:
            window.popleft()
        return sum(load for _, load in window) / len(window)
