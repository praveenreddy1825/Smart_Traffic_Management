"""
Crowd & Pedestrian Detection & Safety Analyzer.

Analyzes detected pedestrians:
  - Tracks live pedestrian headcount in view and cumulative crossings.
  - Measures crowd density level (SPARSE, MODERATE, CROWDED, SURGE_RISK).
  - Identifies pedestrian clusters / gathering hotspots (pairwise spatial proximity).
  - Determines crosswalk occupancy and generates pedestrian walk phase recommendations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.detection.vehicle_detector import Detection

CROWD_SPARSE = "SPARSE"
CROWD_MODERATE = "MODERATE"
CROWD_CROWDED = "CROWDED"
CROWD_SURGE = "SURGE_RISK"


@dataclass
class CrowdReport:
    """Real-time crowd and pedestrian safety assessment."""
    timestamp_sec: float
    headcount: int                       # Pedestrians currently in view
    cumulative_pedestrians: int          # Unique tracked persons counted
    density_level: str                   # SPARSE / MODERATE / CROWDED / SURGE_RISK
    cluster_count: int                   # Number of dense pedestrian clusters
    crosswalk_occupied: bool             # Whether pedestrians are in the road/crossing area
    safety_alert: Optional[str] = None   # Actionable alert message
    recommendation: Optional[str] = None # Traffic signal recommendation for pedestrians


class CrowdAnalyzer:
    """Analyzes pedestrian counts, crowd density, and safety states from detections."""

    def __init__(
        self,
        sparse_threshold: int = 4,
        moderate_threshold: int = 10,
        crowded_threshold: int = 20,
        cluster_distance_pixels: float = 75.0,
        min_frames_to_count: int = 3,
        forget_after_frames: int = 45,
    ):
        self.sparse_threshold = sparse_threshold
        self.moderate_threshold = moderate_threshold
        self.crowded_threshold = crowded_threshold
        self.cluster_distance_pixels = cluster_distance_pixels
        self.min_frames_to_count = min_frames_to_count
        self.forget_after_frames = forget_after_frames

        self.tracked_persons: Dict[int, dict] = {}
        self.counted_ids: set = set()
        self.cumulative_count: int = 0
        self.last_report: Optional[CrowdReport] = None

    def reset(self):
        """Reset all tracking and cumulative counts for a new video clip."""
        self.tracked_persons.clear()
        self.counted_ids.clear()
        self.cumulative_count = 0
        self.last_report = None

    def update(self, detections: List[Detection], frame_index: int, timestamp_sec: float) -> CrowdReport:
        """Process detected persons in the current frame and generate crowd report."""
        person_detections = [d for d in detections if d.class_name == "person"]
        headcount = len(person_detections)

        # Update tracking and cumulative unique person counts
        for d in person_detections:
            t_id = d.track_id
            if t_id is not None:
                if t_id not in self.tracked_persons:
                    self.tracked_persons[t_id] = {
                        "first_seen": frame_index,
                        "last_seen": frame_index,
                        "frames_seen": 1,
                        "center": d.center,
                    }
                else:
                    self.tracked_persons[t_id]["last_seen"] = frame_index
                    self.tracked_persons[t_id]["frames_seen"] += 1
                    self.tracked_persons[t_id]["center"] = d.center

                if t_id not in self.counted_ids and self.tracked_persons[t_id]["frames_seen"] >= self.min_frames_to_count:
                    self.counted_ids.add(t_id)
                    self.cumulative_count += 1

        # Forget stale tracks
        stale_ids = [
            t_id for t_id, info in self.tracked_persons.items()
            if frame_index - info["last_seen"] > self.forget_after_frames
        ]
        for t_id in stale_ids:
            del self.tracked_persons[t_id]

        # Determine crowd density level
        if headcount < self.sparse_threshold:
            density_level = CROWD_SPARSE
        elif headcount < self.moderate_threshold:
            density_level = CROWD_MODERATE
        elif headcount < self.crowded_threshold:
            density_level = CROWD_CROWDED
        else:
            density_level = CROWD_SURGE

        # Cluster detection (pedestrians in close proximity)
        cluster_count = self._detect_clusters(person_detections)

        # Crosswalk occupancy & safety recommendations
        crosswalk_occupied = headcount > 0
        safety_alert = None
        recommendation = None

        if density_level == CROWD_SURGE:
            safety_alert = "CRITICAL: Crowd surge / high density gathering detected!"
            recommendation = "HOLD VEHICLE TRAFFIC: Prioritize pedestrian clearance and crowd dispersion."
        elif density_level == CROWD_CROWDED:
            safety_alert = "WARNING: Heavy pedestrian congestion at crossing zone."
            recommendation = "EXTEND PEDESTRIAN WALK PHASE: +15s walk interval recommended."
        elif density_level == CROWD_MODERATE:
            safety_alert = f"ACTIVE CROSSING: {headcount} pedestrians navigating crosswalk."
            recommendation = "TRIGGER PEDESTRIAN WALK PHASE: Standard 12s crossing interval."
        else:
            recommendation = "Crosswalk clear or low foot-traffic. Standard vehicle cycle maintained."

        report = CrowdReport(
            timestamp_sec=timestamp_sec,
            headcount=headcount,
            cumulative_pedestrians=self.cumulative_count,
            density_level=density_level,
            cluster_count=cluster_count,
            crosswalk_occupied=crosswalk_occupied,
            safety_alert=safety_alert,
            recommendation=recommendation,
        )
        self.last_report = report
        return report

    def _detect_clusters(self, person_detections: List[Detection]) -> int:
        """Count distinct clusters where 2+ pedestrians are within cluster_distance_pixels."""
        if len(person_detections) < 2:
            return 0

        centers = np.array([d.center for d in person_detections])
        n = len(centers)
        visited = set()
        clusters = 0

        dist_sq = self.cluster_distance_pixels ** 2

        for i in range(n):
            if i in visited:
                continue
            # Find neighbors
            diffs = centers - centers[i]
            dists = np.sum(diffs ** 2, axis=1)
            neighbors = np.where(dists < dist_sq)[0]

            if len(neighbors) >= 2:
                clusters += 1
                for idx in neighbors:
                    visited.add(idx)

        return clusters

