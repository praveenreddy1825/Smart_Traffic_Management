"""
Module 4 - Traffic Analysis.

Turns the counts from Module 3 into a traffic picture:

    analyzer = TrafficAnalyzer.from_config(config)
    report = analyzer.update(counting_result, frame.timestamp_sec)
    report.level          # "LOW" / "MEDIUM" / "HIGH"
    print(report.summary_text())

How the level is decided (all values in config.yaml -> traffic):
  1. load    = vehicles visible NOW, each weighted by size (bus 2.5, motorcycle 0.5, ...)
  2. density = average load over the last few seconds of video (smooths flicker)
  3. level   = compare density with two PROTOTYPE thresholds (medium, high)

The thresholds are heuristic and camera-specific: they must be calibrated
from measured densities of real normal / congested footage for each camera.
"""

from collections import deque
from dataclasses import dataclass

LOW, MEDIUM, HIGH = "LOW", "MEDIUM", "HIGH"

# Plural labels for the printed report
CLASS_LABELS = {"car": "Cars", "motorcycle": "Motorcycles", "bus": "Buses", "truck": "Trucks"}


@dataclass
class TrafficReport:
    timestamp_sec: float
    total_unique: int          # unique vehicles counted so far (Module 3)
    unique_by_class: dict
    visible_now: int           # vehicles in the current frame
    visible_by_class: dict
    load_now: float            # weighted vehicles in the current frame
    density: float             # load averaged over the density window
    level: str                 # LOW / MEDIUM / HIGH
    line_crossings: int
    flow_per_minute: float     # line crossings per minute (recent window)

    def summary_text(self, class_names=("car", "motorcycle", "bus", "truck")):
        lines = [f"Total vehicles: {self.total_unique}", ""]
        for name in class_names:
            lines.append(f"{CLASS_LABELS.get(name, name.title())}: "
                         f"{self.unique_by_class.get(name, 0)}")
        lines += ["", f"Vehicles in view now: {self.visible_now}",
                  f"Traffic density: {self.density:.1f} (weighted vehicles, "
                  f"averaged)", f"Traffic Level: {self.level}"]
        return "\n".join(lines)


def classify_level(density, medium_threshold, high_threshold):
    """LOW below medium, MEDIUM from medium up to high, HIGH from high upwards."""
    if density >= high_threshold:
        return HIGH
    if density >= medium_threshold:
        return MEDIUM
    return LOW


def weighted_load(visible_by_class, class_weights):
    """Sum of visible vehicles, each multiplied by its class weight (default 1.0)."""
    return sum(count * class_weights.get(name, 1.0) for name, count in visible_by_class.items())


class TrafficAnalyzer:
    def __init__(self, class_weights=None, density_window_seconds=3.0,
                 medium_threshold=8.0, high_threshold=15.0, flow_window_seconds=60.0):
        if medium_threshold >= high_threshold:
            raise ValueError("traffic thresholds: 'medium' must be smaller than 'high'")
        self.class_weights = dict(class_weights or {})
        self.density_window_seconds = density_window_seconds
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.flow_window_seconds = flow_window_seconds
        self.reset()

    @classmethod
    def from_config(cls, config):
        settings = config["traffic"]
        thresholds = settings["level_thresholds"]
        return cls(
            class_weights=settings.get("class_weights"),
            density_window_seconds=settings.get("density_window_seconds", 3.0),
            medium_threshold=thresholds["medium"],
            high_threshold=thresholds["high"],
            flow_window_seconds=settings.get("flow_window_seconds", 60.0),
        )

    def reset(self):
        self._recent_loads = deque()        # (timestamp, load) inside the density window
        self._crossing_history = deque()    # (timestamp, total crossings so far)

    def update(self, counting_result, timestamp_sec):
        """Feed one frame's CountingResult (Module 3) and its video timestamp."""
        load_now = weighted_load(counting_result.visible_by_class, self.class_weights)

        # Density: average of the loads seen in the last density_window_seconds
        self._recent_loads.append((timestamp_sec, load_now))
        while timestamp_sec - self._recent_loads[0][0] > self.density_window_seconds:
            self._recent_loads.popleft()
        density = sum(load for _, load in self._recent_loads) / len(self._recent_loads)

        return TrafficReport(
            timestamp_sec=timestamp_sec,
            total_unique=counting_result.total_unique,
            unique_by_class=dict(counting_result.unique_by_class),
            visible_now=counting_result.visible_now,
            visible_by_class=dict(counting_result.visible_by_class),
            load_now=load_now,
            density=density,
            level=classify_level(density, self.medium_threshold, self.high_threshold),
            line_crossings=counting_result.line_crossings,
            flow_per_minute=self._flow_per_minute(counting_result.line_crossings, timestamp_sec),
        )

    def _flow_per_minute(self, total_crossings, timestamp_sec):
        """Line crossings in the recent window, scaled to vehicles per minute."""
        self._crossing_history.append((timestamp_sec, total_crossings))
        while timestamp_sec - self._crossing_history[0][0] > self.flow_window_seconds:
            self._crossing_history.popleft()
        oldest_time, oldest_total = self._crossing_history[0]
        elapsed = timestamp_sec - oldest_time
        if elapsed <= 0:
            return 0.0
        return (total_crossings - oldest_total) * 60.0 / elapsed
