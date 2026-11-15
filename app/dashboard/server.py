"""
High-Performance Presentation Server - Scenario Spotlight & Adaptive Multimodal AI.

Serves:
  1. High-FPS Annotated Single Video Feed (/video_feed)
  2. Live Telemetry API (/api/status) with dual Vehicle & Crowd analytics
  3. Scenario Catalog & Switching API (/api/scenarios, /api/scenarios/select)
  4. Video Upload API (/api/scenarios/upload)
  5. Analysis Mode Switcher (/api/control/mode)
"""

from contextlib import asynccontextmanager
import asyncio
import base64
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.analysis.crowd_analyzer import CrowdAnalyzer, CrowdReport
from app.audio.siren_classifier import SirenClassifier
from app.audio.siren_smoother import AudioEmergencyState, SirenSmoother
from app.decision.unified_decision_engine import (
    PHASE_EMERGENCY_CLEARANCE,
    PHASE_GREEN,
    PHASE_PEDESTRIAN_WALK,
    PHASE_RED,
    STATUS_CONGESTION_PRIORITY,
    STATUS_EMERGENCY_PREEMPTION,
    STATUS_NORMAL,
    STATUS_PEDESTRIAN_CROSSING,
    UnifiedDecision,
    UnifiedDecisionEngine,
)
from app.detection.vehicle_detector import Detection, VehicleDetector
from app.tracking.vehicle_counter import VehicleCounter
from app.traffic.traffic_analyzer import TrafficAnalyzer, TrafficReport
from app.utils.config import load_config, resolve_path
from app.utils.device import get_device
from app.utils.logger import get_logger, setup_logging
from app.video.scenario_manager import ScenarioManager, ScenarioMeta

logger = get_logger("dashboard_server")

# Class color palette (BGR)
CLASS_COLORS = {
    "car": (0, 220, 0),          # Green
    "motorcycle": (255, 200, 0), # Cyan
    "bicycle": (255, 255, 0),    # Light Cyan
    "bus": (0, 165, 255),        # Orange
    "truck": (0, 90, 255),       # Amber-Red
    "person": (0, 215, 255),     # Vibrant Gold / Yellow for pedestrians
}


class ScenarioSelectRequest(BaseModel):
    scenario_id: str


class ModeControlRequest(BaseModel):
    mode: str = "auto"  # auto / vehicle / crowd / hybrid


class SpeedControlRequest(BaseModel):
    stride: int = 1


class InjectControlRequest(BaseModel):
    vehicle_type: str = "Ambulance"
    has_emergency: bool = True


class PipelineManager:
    """Coordinates single-stream scenario decoding, dual AI detection, and multimodal fusion."""

    def __init__(self, config: dict):
        self.config = config
        self.running = False
        self.stop_requested = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.frame_counter: int = 0

        # Scenario & Stream Manager
        videos_base_dir = resolve_path(config.get("paths", {}).get("videos_dir", "data/videos"))
        self.scenario_manager = ScenarioManager(videos_base_dir)

        # AI Models
        self.detector = VehicleDetector.from_config(config)
        self.vehicle_classes = list(config.get("detection", {}).get("vehicle_classes", ["car", "bus", "truck", "motorcycle"]))
        self.classifier = SirenClassifier.from_config(config)
        self.smoother = SirenSmoother.from_config(config)

        # Analyzers
        self.counter = VehicleCounter(min_frames_to_count=3, forget_after_frames=60)
        self.traffic_analyzer = TrafficAnalyzer(class_weights=config.get("traffic", {}).get("class_weights"))
        self.crowd_analyzer = CrowdAnalyzer(
            sparse_threshold=config.get("crowd", {}).get("sparse_threshold", 4),
            moderate_threshold=config.get("crowd", {}).get("moderate_threshold", 10),
            crowded_threshold=config.get("crowd", {}).get("crowded_threshold", 20),
        )
        self.decision_engine = UnifiedDecisionEngine()

        # Operational state
        self.active_mode: str = "auto"   # auto / vehicle / crowd / hybrid
        self.frame_stride: int = 1
        self.injected_emergency: bool = False
        self.injected_label: Optional[str] = None

        self.latest_frame: bytes = b""
        self.latest_status: Dict = {}
        self.fps: float = 0.0

        # Event audit log ring buffer
        self.event_log: List[Dict] = []
        self._last_phase: str = ""
        self._last_status: str = ""
        self._last_scenario_id: str = ""
        self._add_event("SYSTEM", "Traffic Operations Center initialized. Multimodal pipeline ready.", "INFO")

        # Load initial scenario
        self._load_initial_scenario()

    def _add_event(self, source: str, message: str, level: str = "INFO"):
        """Record a timestamped event into the circular audit log buffer."""
        now_str = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
        entry = {
            "timestamp": now_str,
            "source": source,
            "message": message,
            "level": level
        }
        self.event_log.append(entry)
        if len(self.event_log) > 50:
            self.event_log.pop(0)

    def _load_initial_scenario(self):
        scenarios = self.scenario_manager.scan_scenarios()
        if scenarios:
            # Prefer a vehicles scenario first
            first_choice = next((s for s in scenarios if s.category == "vehicles"), scenarios[0])
            self.scenario_manager.load_scenario(first_choice.id)
            self._last_scenario_id = first_choice.id
            self._add_event("SOURCE", f"Loaded initial feed: {first_choice.name} ({first_choice.category.upper()})", "INFO")
            logger.info("Loaded default presentation scenario: %s", first_choice.name)

    def select_scenario(self, scenario_id: str):
        """Switch active scenario clip and reset counters."""
        with self.lock:
            self.scenario_manager.load_scenario(scenario_id)
            self.detector.reset_tracking()
            self.counter.reset()
            self.traffic_analyzer.reset()
            self.crowd_analyzer.reset()
            self.smoother.reset()
            self._last_scenario_id = scenario_id
            scenario = self.scenario_manager.active_scenario
            sname = scenario.name if scenario else scenario_id
            self._add_event("SOURCE", f"Switched CCTV feed: {sname}", "INFO")
            logger.info("Switched to scenario: %s", scenario_id)

    def set_mode(self, mode: str):
        with self.lock:
            if mode in ("auto", "vehicle", "crowd", "hybrid"):
                self.active_mode = mode
                self._add_event("CONTROL", f"Analysis focus mode set to: {mode.upper()}", "INFO")

    def set_speed(self, stride: int):
        with self.lock:
            self.frame_stride = max(1, min(6, stride))
            self._add_event("CONTROL", f"Playback speed stride set to: {self.frame_stride}x", "INFO")

    def set_injected_emergency(self, active: bool, label: Optional[str] = "Ambulance"):
        with self.lock:
            self.injected_emergency = active
            self.injected_label = label if active else None
            if active:
                self._add_event("EMERGENCY", f"Simulated {label} beacon injection triggered! Preemption active.", "ALERT")
            else:
                self._add_event("EMERGENCY", "Simulated beacon override cleared. Returning to adaptive cycle.", "INFO")

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Presentation pipeline worker thread started.")

    def stop(self):
        self.stop_requested.set()
        self.running = False
        with self.condition:
            self.condition.notify_all()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        with self.lock:
            if self.scenario_manager.active_stream:
                self.scenario_manager.active_stream.close()
        logger.info("Presentation pipeline worker thread stopped.")

    def _run_loop(self):
        latest_audio_state = AudioEmergencyState(
            state="NO_SIREN", emergency_active=False, window_is_siren=False,
            window_score=0.0, siren_windows_in_history=0, history_length=5,
            smoothed_confidence=0.0, siren_type=None, event_started_sec=None,
            event_duration_sec=0.0, timestamp_sec=0.0, events_so_far=0,
        )
        latest_dbfs = -60.0
        fps_timestamps = []

        while self.running:
            t0 = time.perf_counter()

            try:
                # 1. Read frame and optional audio window from active scenario stream (thread-safe)
                with self.lock:
                    stream = self.scenario_manager.active_stream
                    if not stream or not stream.is_opened:
                        time.sleep(0.04)
                        continue
                    frame, audio_win = stream.read(stride=self.frame_stride)

                if frame is None:
                    time.sleep(0.02)
                    continue

                # 2. Process audio window if newly ready
                if audio_win:
                    try:
                        latest_dbfs = audio_win.rms_dbfs
                        pred = self.classifier.predict(audio_win)
                        latest_audio_state = self.smoother.update(pred)
                    except Exception as err:
                        logger.debug("Audio model error: %s", err)

                # Determine effective analysis mode
                with self.lock:
                    scenario_meta = self.scenario_manager.active_scenario
                    active_mode = self.active_mode
                    has_visual_emerg = self.injected_emergency or (
                        scenario_meta is not None and scenario_meta.category == "emergency"
                    )
                    emerg_label = self.injected_label or ("Ambulance" if has_visual_emerg else None)

                effective_mode = active_mode
                if active_mode == "auto":
                    if scenario_meta and scenario_meta.category == "crowd":
                        effective_mode = "crowd"
                    elif scenario_meta and scenario_meta.category == "vehicles":
                        effective_mode = "vehicle"
                    elif scenario_meta and scenario_meta.category == "emergency":
                        effective_mode = "emergency"

                # 3. Vision Detection & ByteTrack tracking (with crowd multi-scale boost)
                is_crowd_scene = (effective_mode == "crowd") or (scenario_meta is not None and scenario_meta.category == "crowd")
                detections = self.detector.track(frame.image, crowd_mode=is_crowd_scene)

                # Separate vehicles and pedestrians
                veh_detections = [d for d in detections if d.class_name in self.vehicle_classes]
                crowd_detections = [d for d in detections if d.class_name == "person"]

                # 4. Update Analytics
                counting_res = self.counter.update(veh_detections, frame.index)
                traffic_rep = self.traffic_analyzer.update(counting_res, frame.timestamp_sec)
                crowd_rep = self.crowd_analyzer.update(
                    detections, frame.index, frame.timestamp_sec, frame_shape=frame.image.shape
                )

                # 5. Multimodal Decision Engine
                if active_mode == "auto" and effective_mode not in ("crowd", "vehicle", "emergency"):
                    effective_mode = "crowd" if crowd_rep.headcount > traffic_rep.visible_now else "vehicle"

                decision = self.decision_engine.evaluate(
                    vehicle_report=traffic_rep,
                    crowd_report=crowd_rep,
                    audio_state=latest_audio_state,
                    active_mode=effective_mode,
                    has_visual_emergency=has_visual_emerg,
                    emergency_label=emerg_label,
                )

                # Check state transitions for Operations Audit Log
                if decision.signal_phase != self._last_phase:
                    self._last_phase = decision.signal_phase
                    lvl = "WARN" if decision.signal_phase == "RED" else "INFO"
                    self._add_event("SIGNAL", f"Phase transition -> {decision.signal_phase} ({decision.recommended_duration_sec}s allocated)", lvl)
                if decision.system_status != self._last_status:
                    self._last_status = decision.system_status
                    lvl = "ALERT" if "EMERGENCY" in decision.system_status else ("WARN" if "CONGESTION" in decision.system_status else "INFO")
                    self._add_event("INTERLOCK", f"State: {decision.system_status} - {decision.headline}", lvl)

                # 6. Annotate frame for presentation (authentic CCTV surveillance look)
                annotated_img = self._annotate_frame(
                    frame.image, detections, traffic_rep, crowd_rep, decision, effective_mode, stream.has_audio
                )

                # 7. Compress frame to JPEG
                ok, buf = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                frame_bytes = buf.tobytes() if ok else b""

                # 8. Measure FPS
                now = time.perf_counter()
                fps_timestamps.append(now)
                if len(fps_timestamps) > 15:
                    fps_timestamps.pop(0)
                if len(fps_timestamps) >= 2:
                    self.fps = round((len(fps_timestamps) - 1) / (fps_timestamps[-1] - fps_timestamps[0]), 1)

                # 9. Build rich JSON telemetry
                telemetry = {
                    "timestamp": round(now, 2),
                    "fps": self.fps,
                    "device": self.detector.device.upper(),
                    "active_mode": active_mode,
                    "effective_mode": effective_mode,
                    "playback_stride": self.frame_stride,
                    "scenario": {
                        "id": scenario_meta.id if scenario_meta else "none",
                        "name": scenario_meta.name if scenario_meta else "None",
                        "file_name": scenario_meta.file_name if scenario_meta else "",
                        "category": scenario_meta.category if scenario_meta else "vehicles",
                        "has_audio": stream.has_audio,
                        "description": scenario_meta.description if scenario_meta else "",
                    },
                    "vehicles": {
                        "visible_now": traffic_rep.visible_now,
                        "total_unique": traffic_rep.total_unique,
                        "by_class": traffic_rep.visible_by_class,
                        "level": traffic_rep.level,
                        "density": round(traffic_rep.density, 1),
                        "flow_rate_per_min": round(getattr(traffic_rep, "flow_per_minute", 0.0), 1),
                    },
                    "crowd": {
                        "headcount": crowd_rep.headcount,
                        "cumulative": crowd_rep.cumulative_pedestrians,
                        "density_level": crowd_rep.density_level,
                        "estimated_total": crowd_rep.estimated_total_crowd,
                        "density_ratio": crowd_rep.crowd_density_ratio,
                        "cluster_count": crowd_rep.cluster_count,
                        "crosswalk_occupied": crowd_rep.crosswalk_occupied,
                        "safety_alert": crowd_rep.safety_alert,
                        "recommendation": crowd_rep.recommendation,
                    },
                    "audio": {
                        "state": latest_audio_state.state,
                        "emergency_active": latest_audio_state.emergency_active,
                        "confidence": round(latest_audio_state.smoothed_confidence, 2),
                        "siren_type": latest_audio_state.siren_type,
                        "loudness_dbfs": round(latest_dbfs, 1),
                        "has_audio_track": stream.has_audio,
                    },
                    "decision": {
                        "system_status": decision.system_status,
                        "signal_phase": decision.signal_phase,
                        "recommended_duration_sec": decision.recommended_duration_sec,
                        "headline": decision.headline,
                        "action": decision.action,
                        "recommendation": decision.recommendation,
                        "reasons": decision.reasons,
                    },
                    "events": list(self.event_log[-20:]),
                }

                # Update cache safely and notify all waiting MJPEG streamers
                with self.condition:
                    self.latest_frame = frame_bytes
                    self.latest_status = telemetry
                    self.frame_counter += 1
                    self.condition.notify_all()

            except Exception as loop_err:
                logger.error("Error in presentation pipeline loop: %s", loop_err, exc_info=True)
                time.sleep(0.05)

            # Throttle cycle for smooth ~30 FPS operation
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            target_delay = max(0.003, (1000.0 / 30.0 - elapsed_ms) / 1000.0)
            time.sleep(target_delay)

    def _annotate_frame(
        self,
        image: np.ndarray,
        detections: List[Detection],
        traffic_rep: TrafficReport,
        crowd_rep: CrowdReport,
        decision: UnifiedDecision,
        effective_mode: str,
        has_audio: bool,
    ) -> np.ndarray:
        """Render high-clarity bounding boxes, track IDs, and CCTV OSD on the image."""
        h, w = image.shape[:2]

        # 1. Partition Detections into Vehicles and Pedestrians/Crowd
        vehicle_detections = [d for d in detections if d.class_name != "person"]
        person_detections = [d for d in detections if d.class_name == "person"]
        is_crowd_scene = (effective_mode == "crowd") or (crowd_rep.density_level in ("SURGE_RISK", "CROWDED")) or (len(person_detections) >= 8)

        # 1a. Always render Vehicle Bounding Boxes with class colors and tracking IDs
        for d in vehicle_detections:
            x1, y1, x2, y2 = (int(v) for v in d.bbox)
            color = CLASS_COLORS.get(d.class_name, (200, 200, 200))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            id_str = f" #{d.track_id}" if d.track_id else ""
            label = f"{d.class_name.upper()}{id_str} {d.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(image, (x1, max(y1 - 18, 0)), (x1 + lw + 6, max(y1, 18)), (15, 18, 24), -1)
            cv2.rectangle(image, (x1, max(y1 - 18, 0)), (x1 + lw + 6, max(y1, 18)), color, 1)
            cv2.putText(image, label, (x1 + 3, max(y1 - 4, 13)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # 1b. Crowd Presentation: Thermal Density Heatmap + Tactical Dots vs. Individual Boxes
        if is_crowd_scene and person_detections:
            # Generate high-performance Gaussian Thermal Density Heatmap
            scale = 4
            dh, dw = max(1, h // scale), max(1, w // scale)
            density_grid = np.zeros((dh, dw), dtype=np.float32)

            head_coords = []
            for d in person_detections:
                x1, y1, x2, y2 = d.bbox
                # Localize human head position (~top 20% center of detection box)
                hx = int(round((x1 + x2) / 2.0))
                hy = int(round(y1 + (y2 - y1) * 0.20))
                head_coords.append((hx, hy))
                gx = int(hx / scale)
                gy = int(hy / scale)
                if 0 <= gx < dw and 0 <= gy < dh:
                    density_grid[gy, gx] += 1.0

            # Compute smooth continuous thermal distribution
            k_size = int(max(15, min(dh, dw) // 10)) | 1
            density_blurred = cv2.GaussianBlur(density_grid, (k_size, k_size), sigmaX=k_size / 3.2)

            max_val = float(np.max(density_blurred))
            if max_val > 0.05:
                norm = np.clip(density_blurred / (max_val * 0.72 + 1e-6), 0.0, 1.0)
                norm_u8 = (norm * 255).astype(np.uint8)
                heatmap_rgb = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
                heatmap_full = cv2.resize(heatmap_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                density_mask = cv2.resize(norm_u8, (w, h), interpolation=cv2.INTER_LINEAR)

                # Translucent thermal blend only on crowd density zone (surroundings untouched)
                active_mask = density_mask > 22
                if np.any(active_mask):
                    alpha = 0.38
                    blended = cv2.addWeighted(heatmap_full, alpha, image, 1.0 - alpha, 0)
                    image[active_mask] = blended[active_mask]

            # Render tactical glowing head dots (clean point tracking without box clutter)
            for hx, hy in head_coords:
                if 0 <= hx < w and 0 <= hy < h:
                    cv2.circle(image, (hx, hy), 3, (0, 230, 255), -1, cv2.LINE_AA)
                    cv2.circle(image, (hx, hy), 6, (0, 140, 255), 1, cv2.LINE_AA)

            # Floating Tactical Hotspot Tag above primary crowd centroid
            if len(head_coords) >= 12:
                xs = [pt[0] for pt in head_coords]
                ys = [pt[1] for pt in head_coords]
                cx, cy = int(np.mean(xs)), int(min(ys))
                hotspot_tag = f"CROWD DENSITY HOTSPOT [{crowd_rep.density_level} · ~{crowd_rep.estimated_total_crowd} PAX]"
                (tw, th), _ = cv2.getTextSize(hotspot_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                tag_x = max(10, min(cx - tw // 2, w - tw - 14))
                tag_y = max(35, cy - 14)
                cv2.rectangle(image, (tag_x, tag_y - th - 6), (tag_x + tw + 8, tag_y + 4), (10, 14, 22), -1)
                cv2.rectangle(image, (tag_x, tag_y - th - 6), (tag_x + tw + 8, tag_y + 4), (0, 165, 255), 1)
                cv2.putText(image, hotspot_tag, (tag_x + 4, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA)
        else:
            # Sparse Pedestrian Crossing: Draw standard bounding boxes
            for d in person_detections:
                x1, y1, x2, y2 = (int(v) for v in d.bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 220), 2)
                id_str = f" #{d.track_id}" if d.track_id else ""
                label = f"PERSON{id_str} {d.confidence:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                cv2.rectangle(image, (x1, max(y1 - 18, 0)), (x1 + lw + 6, max(y1, 18)), (15, 18, 24), -1)
                cv2.rectangle(image, (x1, max(y1 - 18, 0)), (x1 + lw + 6, max(y1, 18)), (0, 220, 220), 1)
                cv2.putText(image, label, (x1 + 3, max(y1 - 4, 13)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

        # 2. Authentic CCTV Top OSD Strip
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S UTC")
        osd_left = f"CAM-01 [APPROACH FEED] | {ts_str} | AI: YOLOv11+ByteTrack"
        cv2.rectangle(image, (0, 0), (w, 24), (10, 14, 20), -1)
        cv2.line(image, (0, 24), (w, 24), (40, 50, 70), 1)
        cv2.putText(image, osd_left, (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 225), 1)

        # Top-Right Audio Indicator (Text only, no emoji)
        audio_text = "AUDIO: 16kHz MONO (ON)" if has_audio else "AUDIO: MUTE"
        audio_color = (0, 220, 100) if has_audio else (140, 150, 160)
        cv2.putText(image, audio_text, (w - 200, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, audio_color, 1)

        # 3. Operations Status Badge (Upper Left)
        is_crowd_scene = (effective_mode == "crowd") or (crowd_rep.density_level == "SURGE_RISK")
        if decision.system_status == STATUS_EMERGENCY_PREEMPTION:
            badge_bg = (20, 20, 200)
            badge_text = "EMERGENCY PREEMPTION: ARTERIAL CORRIDOR CLEAR"
        elif crowd_rep.density_level == "SURGE_RISK":
            badge_bg = (30, 30, 220)
            badge_text = f"CRITICAL CROWD SURGE (EST. ~{crowd_rep.estimated_total_crowd} PERSONS) - CROWD CLEARANCE ACTIVE"
        elif is_crowd_scene or crowd_rep.headcount > 0:
            badge_bg = (180, 100, 10)
            badge_text = f"CROWD ESTIMATION ACTIVE: {crowd_rep.density_level} (EST. ~{crowd_rep.estimated_total_crowd} PERSONS)"
        elif decision.system_status == STATUS_PEDESTRIAN_CROSSING:
            badge_bg = (180, 100, 10)
            badge_text = f"CROWD ESTIMATION & CLEARANCE ({crowd_rep.density_level})"
        elif decision.system_status == STATUS_CONGESTION_PRIORITY:
            badge_bg = (20, 120, 200)
            badge_text = "ADAPTIVE EXTENSION: HIGH VEHICULAR QUEUE"
        else:
            badge_bg = (20, 90, 30)
            badge_text = f"TRAFFIC CONTROLLER: ADAPTIVE CYCLE (FLOW: {traffic_rep.level})"

        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(image, (10, 32), (10 + bw + 16, 32 + bh + 14), (12, 16, 24), -1)
        cv2.rectangle(image, (10, 32), (10 + bw + 16, 32 + bh + 14), badge_bg, 2)
        cv2.putText(image, badge_text, (18, 32 + bh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 245, 255), 1)

        # 4. Bottom Engineering Telemetry Bar
        crowd_str = (
            f"CROWD ESTIMATION: {crowd_rep.headcount} (EST. GATHERING: ~{crowd_rep.estimated_total_crowd}) | "
            if crowd_rep.estimated_total_crowd > crowd_rep.headcount
            else f"CROWD ESTIMATION: {crowd_rep.headcount} | "
        )

        summary_text = (
            f"VEHICLES IN ZONE: {traffic_rep.visible_now} (TOTAL: {traffic_rep.total_unique}) | "
            f"{crowd_str}"
            f"SIGNAL PHASE: {decision.signal_phase} ({decision.recommended_duration_sec}s)"
        )
        cv2.rectangle(image, (0, h - 26), (w, h), (10, 14, 20), -1)
        cv2.line(image, (0, h - 26), (w, h - 26), (40, 50, 70), 1)
        cv2.putText(image, summary_text, (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 215, 230), 1)

        return image


# --- Global application setup ---
config = load_config()
setup_logging(config)
pipeline = PipelineManager(config)

# Suppress high-frequency status polling noise in console
class _StatusEndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/status" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(_StatusEndpointFilter())

# Patch Uvicorn Server.handle_exit for instantaneous, clean Ctrl+C shutdown
try:
    import uvicorn.server
    _orig_handle_exit = uvicorn.server.Server.handle_exit

    def _instant_handle_exit(self, sig: int, frame) -> None:
        try:
            pipeline.stop()
        except Exception:
            pass
        self.force_exit = True
        return _orig_handle_exit(self, sig, frame)

    uvicorn.server.Server.handle_exit = _instant_handle_exit
except Exception as _patch_err:
    logger.debug("Uvicorn shutdown patch notice: %s", _patch_err)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.start()
    try:
        yield
    finally:
        pipeline.stop()


app = FastAPI(
    title="SmartCity Sentinel - Multi-Modal Traffic & Crowd Presentation Platform",
    version="0.3.0",
    lifespan=lifespan,
)

# Mount static files directory
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        return HTMLResponse("<h2>Dashboard template is loading...</h2>", status_code=200)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/video_feed")
async def video_feed(request: Request):
    """Yield MJPEG boundary stream asynchronously with instant disconnect and graceful shutdown."""
    async def frame_stream():
        last_counter = -1
        try:
            while pipeline.running and not pipeline.stop_requested.is_set():
                if await request.is_disconnected():
                    break

                with pipeline.condition:
                    frame_bytes = pipeline.latest_frame
                    current_counter = pipeline.frame_counter

                if current_counter != last_counter and frame_bytes:
                    last_counter = current_counter
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"

                await asyncio.sleep(0.015)
        except (asyncio.CancelledError, GeneratorExit, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as err:
            logger.debug("Client stream ended: %s", err)

    return StreamingResponse(
        frame_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
        },
    )


@app.get("/api/status")
def get_status():
    with pipeline.lock:
        return JSONResponse(content=pipeline.latest_status)


@app.get("/api/scenarios")
def get_scenarios():
    scenarios = pipeline.scenario_manager.scan_scenarios()
    return JSONResponse(content=[
        {
            "id": s.id,
            "name": s.name,
            "category": s.category,
            "file_name": s.file_name,
            "has_audio": s.has_audio,
            "description": s.description,
        }
        for s in scenarios
    ])


@app.post("/api/scenarios/select")
def select_scenario(req: ScenarioSelectRequest):
    try:
        pipeline.select_scenario(req.scenario_id)
        return {"status": "ok", "scenario_id": req.scenario_id}
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))


@app.post("/api/scenarios/upload")
async def upload_scenario(file: UploadFile = File(...)):
    allowed_exts = (".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png", ".webp")
    if not file.filename.lower().endswith(allowed_exts):
        raise HTTPException(
            status_code=400,
            detail=f"Allowed formats: Video ({'.mp4, .avi, .mov, .mkv'}) or Image ({'.jpg, .jpeg, .png, .webp'}).",
        )

    content = await file.read()
    meta = pipeline.scenario_manager.save_uploaded_clip(file.filename, content)
    pipeline.select_scenario(meta.id)
    return {"status": "ok", "scenario": {"id": meta.id, "name": meta.name, "category": meta.category}}


@app.get("/api/incident/json")
def get_incident_json():
    with pipeline.lock:
        status = dict(pipeline.latest_status)
        scenario = pipeline.scenario_manager.active_scenario

    now_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC")
    incident_id = f"POL-INT402-{int(time.time())}"
    return JSONResponse(content={
        "incident_id": incident_id,
        "timestamp_utc": now_utc,
        "jurisdiction": "Metro Police Department — Traffic Operations Center INT-402",
        "scenario": scenario.name if scenario else "Live Feed",
        "category": scenario.category if scenario else "general",
        "telemetry": status,
    })


@app.get("/api/incident/export", response_class=HTMLResponse)
def export_police_incident_report():
    with pipeline.lock:
        status = dict(pipeline.latest_status)
        frame_bytes = pipeline.latest_frame
        scenario = pipeline.scenario_manager.active_scenario

    now_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC")
    incident_id = f"POL-TMC-402-{int(time.time())}"
    s_name = scenario.name if scenario else "Surveillance Feed Node 402"
    s_cat = (scenario.category if scenario else "GENERAL").upper()

    veh = status.get("vehicles", {})
    crowd = status.get("crowd", {})
    audio = status.get("audio", {})
    decision = status.get("decision", {})
    events = status.get("events", [])

    img_b64 = base64.b64encode(frame_bytes).decode("utf-8") if frame_bytes else ""

    # Classify Incident Code for Police Operations
    sys_status = decision.get("system_status", "NORMAL")
    density = crowd.get("density_level", "SPARSE")
    if "EMERGENCY" in sys_status:
        code = "CODE-3: EMERGENCY ARTERIAL CORRIDOR PREEMPTION"
        code_color = "#ef4444"
        badge_class = "code-emergency"
    elif density == "SURGE_RISK":
        code = "CODE-2: PUBLIC SAFETY CROWD SURGE INTERLOCK"
        code_color = "#f97316"
        badge_class = "code-crowd"
    elif veh.get("level") == "HIGH":
        code = "CODE-1: ARTERIAL CONGESTION MANAGEMENT"
        code_color = "#f59e0b"
        badge_class = "code-traffic"
    else:
        code = "CODE-0: ROUTINE TRAFFIC MONITORING"
        code_color = "#10b981"
        badge_class = "code-normal"

    events_html = "".join(
        f"<tr><td><code>[{e.get('timestamp')}]</code></td><td><strong>{e.get('source')}</strong></td><td>{e.get('message')}</td></tr>"
        for e in events[-8:]
    ) or "<tr><td colspan='3'>No state transitions recorded</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Police Incident Report - {incident_id}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0; padding: 2rem; background: #0b0f17; color: #f8fafc;
    }}
    .container {{ max-width: 960px; margin: 0 auto; background: #111726; border: 1px solid #1e293b; border-radius: 8px; padding: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
    .header {{ display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #1e293b; padding-bottom: 1.5rem; margin-bottom: 1.5rem; }}
    .brand h1 {{ margin: 0; font-size: 1.4rem; color: #38bdf8; letter-spacing: 0.5px; text-transform: uppercase; }}
    .brand p {{ margin: 0.25rem 0 0 0; font-size: 0.85rem; color: #94a3b8; }}
    .meta-box {{ text-align: right; font-family: monospace; font-size: 0.85rem; color: #cbd5e1; }}
    .badge {{ display: inline-block; padding: 0.35rem 0.75rem; border-radius: 4px; font-weight: 700; font-size: 0.85rem; margin-top: 0.5rem; }}
    .code-emergency {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }}
    .code-crowd {{ background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid #f97316; }}
    .code-traffic {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }}
    .code-normal {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }}
    .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem; }}
    .card {{ background: #162032; border: 1px solid #1e293b; border-radius: 6px; padding: 1rem; }}
    .card h3 {{ margin: 0 0 0.5rem 0; font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; }}
    .card .val {{ font-size: 1.8rem; font-weight: 700; font-family: monospace; color: #f1f5f9; }}
    .card .sub {{ font-size: 0.75rem; color: #64748b; margin-top: 0.25rem; }}
    .evidence {{ margin-bottom: 1.5rem; }}
    .evidence h3 {{ margin: 0 0 0.5rem 0; font-size: 0.9rem; color: #cbd5e1; text-transform: uppercase; }}
    .evidence img {{ width: 100%; max-height: 480px; object-fit: contain; background: #000; border: 1px solid #334155; border-radius: 6px; }}
    .advisory-box {{ background: rgba(14, 165, 233, 0.1); border-left: 4px solid #0284c7; padding: 1rem; border-radius: 4px; margin-bottom: 1.5rem; }}
    .advisory-box h4 {{ margin: 0 0 0.25rem 0; color: #38bdf8; font-size: 0.95rem; }}
    .advisory-box p {{ margin: 0; font-size: 0.88rem; color: #e2e8f0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; text-align: left; }}
    th, td {{ padding: 0.5rem; border-bottom: 1px solid #1e293b; }}
    th {{ color: #94a3b8; text-transform: uppercase; font-size: 0.7rem; }}
    .actions {{ display: flex; justify-content: space-between; margin-top: 1.5rem; border-top: 1px solid #1e293b; padding-top: 1rem; }}
    .btn {{ background: #2563eb; color: #fff; border: none; padding: 0.5rem 1rem; border-radius: 4px; font-weight: 600; cursor: pointer; text-decoration: none; display: inline-block; font-size: 0.85rem; }}
    .btn:hover {{ background: #1d4ed8; }}
    .btn-secondary {{ background: #334155; }}
    .btn-secondary:hover {{ background: #475569; }}
    @media print {{
      body {{ background: #fff; color: #000; padding: 0; }}
      .container {{ border: none; box-shadow: none; max-width: 100%; padding: 0; background: #fff; color: #000; }}
      .actions {{ display: none; }}
      .card {{ background: #f8fafc; border: 1px solid #e2e8f0; color: #000; }}
      .card .val {{ color: #000; }}
      .brand h1 {{ color: #000; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="brand">
        <h1>Metropolitan Police Department</h1>
        <p>Traffic Operations Center (TMC) — Autonomous Multimodal Edge Intelligence Incident Log</p>
        <span class="badge {badge_class}">{code}</span>
      </div>
      <div class="meta-box">
        <div><strong>Incident ID:</strong> {incident_id}</div>
        <div><strong>Timestamp:</strong> {now_utc}</div>
        <div><strong>Node:</strong> Intersection INT-402 (Broadway & 5th Ave)</div>
        <div><strong>CCTV Feed:</strong> {s_name} ({s_cat})</div>
      </div>
    </div>

    <div class="evidence">
      <h3>Visual Surveillance & Evidence Still (Annotated)</h3>
      {f'<img src="data:image/jpeg;base64,{img_b64}" alt="CCTV Incident Evidence">' if img_b64 else '<p style="color:#64748b;">No frame available</p>'}
    </div>

    <div class="grid-3">
      <div class="card">
        <h3>Vehicle Traffic Flow</h3>
        <div class="val">{veh.get('visible_now', 0)}</div>
        <div class="sub">Level: <strong>{veh.get('level', 'LOW')}</strong> | Total Passed: {veh.get('total_unique', 0)}</div>
        <div class="sub">Cars: {veh.get('by_class', {}).get('car', 0)} | Buses: {veh.get('by_class', {}).get('bus', 0)} | Trucks: {veh.get('by_class', {}).get('truck', 0)}</div>
      </div>

      <div class="card">
        <h3>Crowd Estimation</h3>
        <div class="val">{crowd.get('headcount', 0)} <span style="font-size:1rem; color:#f87171;">(~{crowd.get('estimated_total', 0)} Est.)</span></div>
        <div class="sub">Density: <strong>{crowd.get('density_level', 'SPARSE')}</strong></div>
        <div class="sub">Clusters: {crowd.get('cluster_count', 0)} | Interlock: <strong>{'ENGAGED' if crowd.get('density_level') == 'SURGE_RISK' else 'MONITORING'}</strong></div>
      </div>

      <div class="card">
        <h3>Acoustic Intelligence</h3>
        <div class="val">{round(audio.get('confidence', 0.0) * 100, 1)}%</div>
        <div class="sub">State: <strong>{audio.get('state', 'NO_SIREN')}</strong></div>
        <div class="sub">Siren Type: {audio.get('siren_type') or 'None detected'} | Level: {audio.get('loudness_dbfs', -60.0)} dBFS</div>
      </div>
    </div>

    <div class="advisory-box">
      <h4>🚨 Police Dispatch Action & Signal Timing Protocol</h4>
      <p><strong>Signal Controller Decision:</strong> {decision.get('headline', 'Standard Cycle Active')}</p>
      <p style="margin-top:0.4rem;"><strong>Recommended Operational Action:</strong> {decision.get('action', 'Hold vehicle traffic on red for clearance.')}</p>
      <p style="margin-top:0.4rem;"><strong>Field Advisory:</strong> {decision.get('recommendation', 'Maintain standard surveillance cycle.')}</p>
    </div>

    <div>
      <h3 style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase;">Recent Operations Audit Trail</h3>
      <table>
        <thead>
          <tr><th>Time</th><th>Source</th><th>Event Message</th></tr>
        </thead>
        <tbody>
          {events_html}
        </tbody>
      </table>
    </div>

    <div class="actions">
      <a href="/" class="btn btn-secondary">← Back to Live CCTV Dashboard</a>
      <div>
        <a href="/api/incident/json" target="_blank" class="btn btn-secondary" style="margin-right:0.5rem;">Download JSON</a>
        <button onclick="window.print()" class="btn">Print / Save Police Dossier (PDF)</button>
      </div>
    </div>
  </div>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.post("/api/control/mode")
def set_mode(req: ModeControlRequest):
    pipeline.set_mode(req.mode)
    return {"status": "ok", "mode": req.mode}


@app.post("/api/control/speed")
def change_playback_speed(req: SpeedControlRequest):
    pipeline.set_speed(req.stride)
    return {"status": "ok", "stride": req.stride}


@app.post("/api/control/inject")
def inject_emergency(req: InjectControlRequest):
    pipeline.set_injected_emergency(req.has_emergency, req.vehicle_type)
    return {"status": "ok", "injected": req.has_emergency, "type": req.vehicle_type}


@app.post("/api/control/clear-inject")
def clear_emergency():
    pipeline.set_injected_emergency(False, None)
    return {"status": "ok", "injected": False}


@app.get("/api/health")
def health():
    return {"status": "healthy", "device": get_device(config.get("device", "auto"))}
