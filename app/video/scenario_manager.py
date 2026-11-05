"""
Scenario Manager.

Discovers, categorizes, and coordinates scenario video clips (Vehicles, Crowd, Emergency),
and manages the active single-video pipeline with embedded audio decoding.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import shutil

from app.utils.config import resolve_path
from app.utils.logger import get_logger
from app.video.scenario_stream import ScenarioStream

logger = get_logger(__name__)

CATEGORY_VEHICLES = "vehicles"
CATEGORY_CROWD = "crowd"
CATEGORY_EMERGENCY = "emergency"
CATEGORY_UPLOAD = "uploads"

FRIENDLY_NAMES = {
    "01_dense_intersection.mp4": "Dense Junction Congestion",
    "02_urban_traffic_flow.mp4": "Urban Traffic Flow (Multi-Class)",
    "01_pedestrian_crossing.mp4": "Active Pedestrian Crosswalk",
    "02_busy_crosswalk_audio.mp4": "Busy Crosswalk with Street Audio",
    "01_ambulance_corridor.mp4": "Emergency Ambulance Corridor",
    "13268870_960_540_30fps.mp4": "High-Volume Arterial Road",
    "12937229_960_540_30fps.mp4": "North City Intersection",
    "13105482_960_540_30fps.mp4": "East Crosswalk & Avenue",
    "14552311-sd_960_540_25fps.mp4": "West Commercial Crosswalk",
    "14552312-hd_1280_720_50fps.mp4": "Emergency Transit Corridor",
}


@dataclass
class ScenarioMeta:
    """Metadata describing a scenario clip."""
    id: str
    name: str
    category: str                   # vehicles / crowd / emergency / upload
    file_name: str
    rel_path: str
    abs_path: str
    has_audio: bool
    description: str


class ScenarioManager:
    """Discovers and switches between scenario clips for panel presentation."""

    def __init__(self, base_videos_dir: Path):
        self.base_dir = Path(base_videos_dir)
        self.active_stream: Optional[ScenarioStream] = None
        self.active_scenario: Optional[ScenarioMeta] = None
        self.upload_dir = self.base_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def scan_scenarios(self) -> List[ScenarioMeta]:
        """Scan directory and return all available categorized scenario videos."""
        scenarios = []

        # 1. Primary categorized subfolders
        category_folders = [
            (CATEGORY_VEHICLES, self.base_dir / "vehicles"),
            (CATEGORY_CROWD, self.base_dir / "crowd"),
            (CATEGORY_EMERGENCY, self.base_dir / "emergency"),
            (CATEGORY_UPLOAD, self.upload_dir),
        ]

        seen_files = set()

        for cat, folder in category_folders:
            if not folder.is_dir():
                continue
            video_files = []
            for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"):
                video_files.extend(folder.glob(ext))
            for video_file in sorted(video_files, key=lambda f: f.name.lower()):
                if video_file.name in seen_files:
                    continue
                seen_files.add(video_file.name)

                has_wav = video_file.with_suffix(".wav").is_file() or video_file.with_suffix(".mp3").is_file()
                name = FRIENDLY_NAMES.get(video_file.name, video_file.stem.replace("_", " ").title())
                s_id = f"{cat}_{video_file.stem}"

                desc = (
                    "Vehicle tracking, counting, and congestion signal extension" if cat == CATEGORY_VEHICLES
                    else ("Pedestrian headcount, crowd density, and crosswalk safety hold" if cat == CATEGORY_CROWD
                          else ("Emergency preemption corridor with audio siren verification" if cat == CATEGORY_EMERGENCY
                                else "User uploaded presentation clip"))
                )

                scenarios.append(ScenarioMeta(
                    id=s_id,
                    name=name,
                    category=cat,
                    file_name=video_file.name,
                    rel_path=str(video_file.relative_to(self.base_dir.parent.parent)),
                    abs_path=str(video_file.resolve()),
                    has_audio=has_wav or cat == CATEGORY_EMERGENCY or "audio" in video_file.name.lower(),
                    description=desc,
                ))

        return scenarios

    def load_scenario(self, scenario_id_or_path: str, max_frame_size: int = 720) -> ScenarioStream:
        """Switch active scenario to the requested clip."""
        all_scenarios = self.scan_scenarios()
        chosen = None

        # Try match by ID
        for s in all_scenarios:
            if s.id == scenario_id_or_path or s.file_name == scenario_id_or_path:
                chosen = s
                break

        # Try match by path
        if chosen is None:
            cand_path = resolve_path(scenario_id_or_path)
            if cand_path.is_file():
                cat = (
                    CATEGORY_EMERGENCY if "emergency" in str(cand_path).lower()
                    else (CATEGORY_CROWD if "crowd" in str(cand_path).lower() else CATEGORY_VEHICLES)
                )
                chosen = ScenarioMeta(
                    id=f"custom_{cand_path.stem}",
                    name=cand_path.stem.replace("_", " ").title(),
                    category=cat,
                    file_name=cand_path.name,
                    rel_path=str(cand_path),
                    abs_path=str(cand_path.resolve()),
                    has_audio=cand_path.with_suffix(".wav").is_file(),
                    description="Custom scenario video",
                )

        if chosen is None:
            # Fall back to first available scenario
            if all_scenarios:
                chosen = all_scenarios[0]
            else:
                raise FileNotFoundError("No scenario video clips available.")

        # Close existing stream
        if self.active_stream:
            self.active_stream.close()

        # Open new stream
        stream = ScenarioStream(
            video_path=Path(chosen.abs_path),
            max_frame_size=max_frame_size,
            loop=True,
        )
        stream.open()

        self.active_stream = stream
        self.active_scenario = chosen
        logger.info("Switched active scenario to: %s (%s)", chosen.name, chosen.category)
        return stream

    def save_uploaded_clip(self, file_name: str, content: bytes) -> ScenarioMeta:
        """Save a newly uploaded MP4 file and return its metadata."""
        safe_name = Path(file_name).name
        target_path = self.upload_dir / safe_name
        with open(target_path, "wb") as f:
            f.write(content)

        logger.info("Saved uploaded clip: %s (%d bytes)", target_path.name, len(content))
        return ScenarioMeta(
            id=f"upload_{target_path.stem}",
            name=f"Custom: {target_path.stem.replace('_', ' ').title()}",
            category=CATEGORY_UPLOAD,
            file_name=target_path.name,
            rel_path=str(target_path.relative_to(self.base_dir.parent.parent)),
            abs_path=str(target_path.resolve()),
            has_audio=False,
            description="Uploaded presentation scenario",
        )

