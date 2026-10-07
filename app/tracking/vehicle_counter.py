"""
Module 3 - Unique vehicle counting from tracked detections.

The tracker (ByteTrack, see VehicleDetector.track) gives every vehicle an ID
that stays the same while the vehicle is visible. This module turns those IDs
into counts WITHOUT double counting:

    Frame 1 -> car ID 1
    Frame 2 -> car ID 1        unique count = 1  (not 4)
    Frame 3 -> car ID 1
    Frame 4 -> car ID 1

Rules:
  1. A track ID is counted ONCE, after it has been seen in at least
     `min_frames_to_count` frames (filters one-frame false detections).
  2. A vehicle's class is the label it received MOST OFTEN (majority vote),
     because YOLO sometimes flickers between e.g. "car" and "truck".
  3. Optional line crossing: when a vehicle's centre moves across a virtual
     line, it is counted once in "line crossings", with its direction.
"""

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class TrackInfo:
    """What we remember about one tracked vehicle."""
    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    frames_seen: int = 0
    class_votes: Counter = field(default_factory=Counter)
    last_center: tuple | None = None
    trail: list = field(default_factory=list)   # recent centre points (for drawing)
    counted: bool = False
    crossed_line: bool = False

    @property
    def class_name(self):
        """Majority-vote class over all frames this vehicle was seen in."""
        return self.class_votes.most_common(1)[0][0]


@dataclass
class CountingResult:
    """Snapshot of all counts after the latest frame."""
    total_unique: int
    unique_by_class: dict
    line_crossings: int
    line_crossings_by_class: dict
    line_crossings_by_direction: dict
    visible_now: int
    visible_by_class: dict


def _side_of_line(point, line_start, line_end):
    """Which side of the line a point is on: >0 one side, <0 other side, 0 on the line.

    This is the 2-D cross product (line direction x vector to the point).
    """
    (x1, y1), (x2, y2), (px, py) = line_start, line_end, point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def _segments_intersect(p1, p2, q1, q2):
    """True if segment p1-p2 (vehicle movement) crosses segment q1-q2 (counting line)."""
    d1 = _side_of_line(p1, q1, q2)
    d2 = _side_of_line(p2, q1, q2)
    d3 = _side_of_line(q1, p1, p2)
    d4 = _side_of_line(q2, p1, p2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


class VehicleCounter:
    def __init__(self, min_frames_to_count=3, forget_after_frames=60,
                 counting_line=None, trail_length=20):
        """
        counting_line: ((x1, y1), (x2, y2)) in PIXELS, or None to disable.
        """
        self.min_frames_to_count = min_frames_to_count
        self.forget_after_frames = forget_after_frames
        self.counting_line = counting_line
        self.trail_length = trail_length
        self.reset()

    @classmethod
    def from_config(cls, config, frame_size):
        """Build from the 'tracking' config. frame_size = (width, height) of processed frames,
        used to convert the line's 0-1 fractions into pixels."""
        settings = config["tracking"]
        return cls(
            min_frames_to_count=settings.get("min_frames_to_count", 3),
            forget_after_frames=settings.get("forget_after_frames", 60),
            counting_line=line_to_pixels(settings.get("counting_line"), frame_size),
        )

    def reset(self):
        self.tracks = {}                        # track_id -> TrackInfo (recent vehicles)
        self._finished_class_counts = Counter() # classes of counted vehicles already forgotten
        self.line_crossings_by_class = Counter()
        self.line_crossings_by_direction = Counter()
        self._visible_now = []

    def update(self, detections, frame_index):
        """Feed the tracked detections of ONE frame. Returns a CountingResult."""
        visible = []
        for detection in detections:
            if detection.track_id is None:
                continue  # untracked boxes cannot be counted reliably
            track = self.tracks.get(detection.track_id)
            if track is None:
                track = TrackInfo(detection.track_id, frame_index, frame_index)
                self.tracks[detection.track_id] = track

            previous_center = track.last_center
            track.last_seen_frame = frame_index
            track.frames_seen += 1
            track.class_votes[detection.class_name] += 1
            track.last_center = detection.center
            track.trail = (track.trail + [detection.center])[-self.trail_length:]

            # Rule 1: count each ID once, after it is seen in enough frames
            if not track.counted and track.frames_seen >= self.min_frames_to_count:
                track.counted = True

            # Rule 3: line crossing (once per vehicle)
            if self.counting_line and previous_center and not track.crossed_line:
                self._check_line_crossing(track, previous_center, detection.center)

            visible.append(track)

        self._visible_now = visible
        self._forget_old_tracks(frame_index)
        return self.result()

    def result(self):
        unique_by_class = Counter(self._finished_class_counts)
        for track in self.tracks.values():
            if track.counted:
                unique_by_class[track.class_name] += 1
        visible_by_class = Counter(track.class_name for track in self._visible_now)
        return CountingResult(
            total_unique=sum(unique_by_class.values()),
            unique_by_class=dict(unique_by_class),
            line_crossings=sum(self.line_crossings_by_class.values()),
            line_crossings_by_class=dict(self.line_crossings_by_class),
            line_crossings_by_direction=dict(self.line_crossings_by_direction),
            visible_now=len(self._visible_now),
            visible_by_class=dict(visible_by_class),
        )

    def _check_line_crossing(self, track, previous_center, current_center):
        line_start, line_end = self.counting_line
        if _segments_intersect(previous_center, current_center, line_start, line_end):
            track.crossed_line = True
            # Direction in plain words: up/down for a horizontal-ish line,
            # left/right for a vertical-ish line (image y grows downwards)
            line_dx = abs(line_end[0] - line_start[0])
            line_dy = abs(line_end[1] - line_start[1])
            if line_dx >= line_dy:
                direction = "down" if current_center[1] > previous_center[1] else "up"
            else:
                direction = "right" if current_center[0] > previous_center[0] else "left"
            self.line_crossings_by_direction[direction] += 1
            self.line_crossings_by_class[track.class_name] += 1

    def _forget_old_tracks(self, frame_index):
        """Remove vehicles not seen for a while; their counts are kept."""
        for track_id in list(self.tracks):
            track = self.tracks[track_id]
            if frame_index - track.last_seen_frame > self.forget_after_frames:
                if track.counted:
                    self._finished_class_counts[track.class_name] += 1
                del self.tracks[track_id]


def line_to_pixels(line_fractions, frame_size):
    """Convert [[x1, y1], [x2, y2]] given as 0-1 fractions into pixel coordinates."""
    if not line_fractions:
        return None
    width, height = frame_size
    (fx1, fy1), (fx2, fy2) = line_fractions
    return ((fx1 * width, fy1 * height), (fx2 * width, fy2 * height))
