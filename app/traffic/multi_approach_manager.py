"""
Module 11A - Multi-Approach Traffic Manager.

Manages four directional approaches (North, South, East, West) at an intersection:
  - Captures video frames from four independent sources (files, webcams, or synthetic).
  - Detects and tracks vehicles on each approach using YOLO + ByteTrack.
  - Computes directional traffic statistics: vehicle counts by class, congestion
    level (LOW/MEDIUM/HIGH), and queue density estimates.
  - Monitors visual emergency vehicle presence per approach.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from app.detection.vehicle_detector import Detection, VehicleDetector
from app.tracking.vehicle_counter import CountingResult, VehicleCounter, line_to_pixels
from app.traffic.traffic_analyzer import HIGH, LOW, MEDIUM, TrafficAnalyzer, TrafficReport
from app.utils.config import resolve_path
from app.utils.logger import get_logger
from app.video.video_input import VideoFrame, VideoInput, VideoSourceError

logger = get_logger(__name__)

APPROACH_DIRECTIONS = ("north", "south", "east", "west")
DEFAULT_APPROACH_NAMES = {
    "north": "North Approach",
    "south": "South Approach",
    "east": "East Approach",
    "west": "West Approach",
}

# Color palette for classes (BGR)
CLASS_COLORS = {
    "car": (0, 255, 0),         # Green
    "motorcycle": (255, 200, 0), # Cyan/Yellow
    "bicycle": (255, 255, 0),    # Cyan
    "bus": (0, 165, 255),       # Orange
    "truck": (0, 100, 255),      # Amber/Red
}


def resolve_approach_source(source):
    """Resolve a source path or scan a directory for the first video file."""
    if source is None:
        return None
    if isinstance(source, int):
        return source
    p = resolve_path(str(source))
    if p.is_dir():
        video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
        for ext in video_extensions:
            files = sorted(p.glob(f"*{ext}"))
            if files:
                return files[0]
        return None
    return p if p.is_file() else None


@dataclass
class ApproachState:
    """Real-time traffic and visual state for one directional approach."""
    direction_id: str                      # "north", "south", "east", "west"
    name: str                              # e.g. "North Approach"
    frame: Optional[VideoFrame]            # latest frame from video
    annotated_image: Optional[np.ndarray]  # frame with bounding boxes & info
    detections: List[Detection]            # vehicles currently detected
    traffic_report: Optional[TrafficReport]# traffic metrics (density, counts, level)
    has_visual_emergency: bool = False     # whether an emergency vehicle is in view
    emergency_label: Optional[str] = None  # e.g. "Ambulance", "Police"


class MultiApproachManager:
    """Coordinates video streams, detection, tracking, and traffic analysis across 4 approaches."""

    def __init__(self, approaches_config: dict,
                 detector: Union[VehicleDetector, Dict[str, VehicleDetector]],
                 traffic_config: dict = None):
        self.approaches_config = approaches_config
        if isinstance(detector, dict):
            self.detectors = detector
            self.detector = next(iter(detector.values())) if detector else None
        else:
            self.detector = detector
            self.detectors = {d: detector for d in APPROACH_DIRECTIONS}

        self.traffic_config = traffic_config or {}

        self.video_inputs: Dict[str, VideoInput] = {}
        self.counters: Dict[str, VehicleCounter] = {}
        self.analyzers: Dict[str, TrafficAnalyzer] = {}
        self.manual_emergency_flags: Dict[str, bool] = {d: False for d in APPROACH_DIRECTIONS}
        self.manual_emergency_labels: Dict[str, Optional[str]] = {d: None for d in APPROACH_DIRECTIONS}

        self.step_counter: int = 0
        self.frame_stride: int = 3
        self.cached_detections: Dict[str, List[Detection]] = {d: [] for d in APPROACH_DIRECTIONS}
        self.cached_reports: Dict[str, Optional[TrafficReport]] = {d: None for d in APPROACH_DIRECTIONS}

        self._setup_subsystems()

    def _setup_subsystems(self):
        """Initialize counters and analyzers for each directional approach."""
        for direction_id in APPROACH_DIRECTIONS:
            cfg = self.approaches_config.get(direction_id, {})
            self.counters[direction_id] = VehicleCounter(
                min_frames_to_count=cfg.get("min_frames_to_count", 5),
                forget_after_frames=cfg.get("forget_after_frames", 60),
                counting_line=None,  # converted to pixel coordinates on first frame
            )
            self.analyzers[direction_id] = TrafficAnalyzer(
                class_weights=self.traffic_config.get("class_weights"),
                density_window_seconds=self.traffic_config.get("density_window_seconds", 3.0),
                medium_threshold=self.traffic_config.get("level_thresholds", {}).get("medium", 8.0),
                high_threshold=self.traffic_config.get("level_thresholds", {}).get("high", 15.0),
                flow_window_seconds=self.traffic_config.get("flow_window_seconds", 60.0),
            )

    @classmethod
    def from_config(cls, config: dict, loop_videos: bool = True):
        """Construct manager using application configuration."""
        intersection_cfg = config.get("intersection", {})
        approaches_cfg = intersection_cfg.get("approaches", {})

        # Default demo clips
        default_videos = {
            "north": "data/videos/13997539_1080_1920_30fps.mp4",
            "south": "data/videos/16922729_1920_1080_30fps.mp4",
            "east": "data/videos/13275193-hd_1920_1080_60fps.mp4",
            "west": "data/videos/14552312-hd_1920_1080_50fps.mp4",
        }

        resolved_approaches = {}
        for d in APPROACH_DIRECTIONS:
            user_app_cfg = approaches_cfg.get(d, {})
            source = user_app_cfg.get("video_source", default_videos.get(d))
            counting_line = user_app_cfg.get("counting_line")
            if d == "north" and counting_line is None:
                counting_line = config.get("tracking", {}).get("counting_line")

            resolved_approaches[d] = {
                "name": user_app_cfg.get("name", DEFAULT_APPROACH_NAMES[d]),
                "video_source": source,
                "counting_line": counting_line,
                "loop_video": loop_videos,
                "max_frame_size": min(720, config.get("video", {}).get("max_frame_size", 720)),
                "min_frames_to_count": config.get("tracking", {}).get("min_frames_to_count", 5),
            }

        # Each approach gets its own detector for independent ByteTrack tracking
        detectors = {d: VehicleDetector.from_config(config) for d in APPROACH_DIRECTIONS}
        return cls(resolved_approaches, detectors, config.get("traffic", {}))

    def open(self):
        """Open all directional video sources."""
        for direction_id, cfg in self.approaches_config.items():
            raw_source = cfg.get("video_source")
            resolved = resolve_approach_source(raw_source)
            if resolved is None:
                logger.info("Approach %s (%s): no video file present yet in folder. Feeds in standby.",
                            direction_id, raw_source)
                continue
            try:
                video = VideoInput(
                    source=resolved,
                    max_frame_size=cfg.get("max_frame_size", 960),
                    realtime_playback=False,
                    loop_video=cfg.get("loop_video", True),
                )
                video.open()
                self.video_inputs[direction_id] = video
                logger.info("Opened approach %s video: %s", direction_id, resolved)
            except VideoSourceError as err:
                logger.warning("Could not open video for %s (%s): %s", direction_id, resolved, err)
        return self

    def close(self):
        """Close all opened video streams."""
        for direction_id, video in list(self.video_inputs.items()):
            video.release()
        self.video_inputs.clear()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def set_manual_emergency(self, direction_id: str, active: bool, label: str = "Ambulance"):
        """Manually trigger or clear a visual emergency on a specific approach."""
        if direction_id in self.manual_emergency_flags:
            self.manual_emergency_flags[direction_id] = active
            self.manual_emergency_labels[direction_id] = label if active else None

    def step(self, stride: Optional[int] = None) -> Dict[str, ApproachState]:
        """Advance one frame on each approach and return the updated states."""
        self.step_counter += 1
        active_stride = stride if stride is not None else self.frame_stride
        states = {}

        # Round-robin interleaved detection:
        # Run YOLO inference on 1 approach per step (or force on manual emergency).
        # This reduces CPU inference time by 75% while keeping all 4 video feeds
        # playing at smooth, real-world traffic speed!
        primary_detect_dir = APPROACH_DIRECTIONS[self.step_counter % len(APPROACH_DIRECTIONS)]

        for direction_id in APPROACH_DIRECTIONS:
            cfg = self.approaches_config.get(direction_id, {})
            name = cfg.get("name", DEFAULT_APPROACH_NAMES.get(direction_id, direction_id.title()))
            video = self.video_inputs.get(direction_id)

            if video is None or not video.is_opened:
                states[direction_id] = ApproachState(
                    direction_id=direction_id,
                    name=name,
                    frame=None,
                    annotated_image=None,
                    detections=[],
                    traffic_report=None,
                    has_visual_emergency=self.manual_emergency_flags.get(direction_id, False),
                    emergency_label=self.manual_emergency_labels.get(direction_id),
                )
                continue

            frame = video.read(stride=active_stride)
            if frame is None:
                states[direction_id] = ApproachState(
                    direction_id=direction_id,
                    name=name,
                    frame=None,
                    annotated_image=None,
                    detections=[],
                    traffic_report=None,
                    has_visual_emergency=self.manual_emergency_flags.get(direction_id, False),
                    emergency_label=self.manual_emergency_labels.get(direction_id),
                )
                continue

            # Run detection if it is this approach's turn or no cached detections yet
            should_detect = (
                direction_id == primary_detect_dir
                or not self.cached_detections.get(direction_id)
                or self.manual_emergency_flags.get(direction_id, False)
            )

            if should_detect:
                detector = self.detectors.get(direction_id, self.detector)
                if hasattr(detector, "track"):
                    detections = detector.track(frame.image)
                else:
                    detections = detector.detect(frame.image)
                    for idx, d in enumerate(detections):
                        if d.track_id is None:
                            d.track_id = idx + 1
                self.cached_detections[direction_id] = detections
            else:
                detections = self.cached_detections.get(direction_id, [])

            # Initialize counting line in pixel coords on first frame if needed
            counter = self.counters[direction_id]
            if counter.counting_line is None and cfg.get("counting_line"):
                h, w = frame.image.shape[:2]
                counter.counting_line = line_to_pixels(cfg.get("counting_line"), (w, h))

            counting_res = counter.update(detections, frame.index)
            analyzer = self.analyzers[direction_id]
            report = analyzer.update(counting_res, frame.timestamp_sec)
            self.cached_reports[direction_id] = report

            has_emerg = self.manual_emergency_flags.get(direction_id, False)
            emerg_label = self.manual_emergency_labels.get(direction_id)

            annotated = self._annotate_frame(frame.image, detections, report, has_emerg, emerg_label)

            states[direction_id] = ApproachState(
                direction_id=direction_id,
                name=name,
                frame=frame,
                annotated_image=annotated,
                detections=detections,
                traffic_report=report,
                has_visual_emergency=has_emerg,
                emergency_label=emerg_label,
            )

        return states

    def _annotate_frame(self, image: np.ndarray, detections: List[Detection],
                        report: Optional[TrafficReport], has_emergency: bool,
                        emergency_label: Optional[str]) -> np.ndarray:
        """Draw detections and traffic info box onto frame."""
        h, w = image.shape[:2]

        # Draw bounding boxes with track IDs
        for d in detections:
            x1, y1, x2, y2 = (int(v) for v in d.bbox)
            color = CLASS_COLORS.get(d.class_name, (0, 255, 0))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            id_str = f" #{d.track_id}" if d.track_id else ""
            label = f"{d.class_name}{id_str} {d.confidence:.2f}"
            cv2.rectangle(image, (x1, max(y1 - 18, 0)), (x1 + len(label) * 8 + 6, max(y1, 18)), (20, 20, 20), -1)
            cv2.putText(image, label, (x1 + 3, max(y1 - 4, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # Highlight emergency banner if present
        if has_emergency:
            cv2.rectangle(image, (0, 0), (w, 36), (0, 0, 220), -1)
            cv2.putText(image, f"EMERGENCY VEHICLE DETECTED: {emergency_label or 'Ambulance'}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Traffic Level Tag and Count at bottom
        if report:
            level_color = (0, 200, 0) if report.level == LOW else ((0, 180, 255) if report.level == MEDIUM else (0, 0, 255))
            cv2.rectangle(image, (8, h - 35), (250, h - 8), (20, 20, 20), -1)
            cv2.putText(image, f"{report.level} ({report.visible_now} vecs in view)", (14, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, level_color, 2)

        return image
