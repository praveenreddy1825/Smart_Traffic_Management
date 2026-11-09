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

                # 3. Vision Detection & ByteTrack tracking (single video stream -> blazing fast!)
                detections = self.detector.track(frame.image)

                # Separate vehicles and pedestrians
                veh_detections = [d for d in detections if d.class_name in self.vehicle_classes]
                crowd_detections = [d for d in detections if d.class_name == "person"]

                # 4. Update Analytics
                counting_res = self.counter.update(veh_detections, frame.index)
                traffic_rep = self.traffic_analyzer.update(counting_res, frame.timestamp_sec)
                crowd_rep = self.crowd_analyzer.update(detections, frame.index, frame.timestamp_sec)

                # 5. Multimodal Decision Engine
                with self.lock:
                    scenario_meta = self.scenario_manager.active_scenario
                    active_mode = self.active_mode
                    has_visual_emerg = self.injected_emergency or (
                        scenario_meta is not None and scenario_meta.category == "emergency"
                    )
                    emerg_label = self.injected_label or ("Ambulance" if has_visual_emerg else None)

                # Determine effective analysis mode
                effective_mode = active_mode
                if active_mode == "auto":
                    if scenario_meta and scenario_meta.category == "crowd":
                        effective_mode = "crowd"
                    elif scenario_meta and scenario_meta.category == "vehicles":
                        effective_mode = "vehicle"
                    elif scenario_meta and scenario_meta.category == "emergency":
                        effective_mode = "emergency"
                    else:
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

        # 1. Draw Bounding Boxes with Class Colors
        for d in detections:
            x1, y1, x2, y2 = (int(v) for v in d.bbox)
            color = CLASS_COLORS.get(d.class_name, (200, 200, 200))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            id_str = f" #{d.track_id}" if d.track_id else ""
            label = f"{d.class_name.upper()}{id_str} {d.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(image, (x1, max(y1 - 18, 0)), (x1 + lw + 6, max(y1, 18)), (15, 18, 24), -1)
            cv2.rectangle(image, (x1, max(y1 - 18, 0)), (x1 + lw + 6, max(y1, 18)), color, 1)
            cv2.putText(image, label, (x1 + 3, max(y1 - 4, 13)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

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
        if decision.system_status == STATUS_EMERGENCY_PREEMPTION:
            badge_bg = (20, 20, 200)
            badge_text = "EMERGENCY PREEMPTION: ARTERIAL CORRIDOR CLEAR"
        elif decision.system_status == STATUS_PEDESTRIAN_CROSSING:
            badge_bg = (180, 100, 10)
            badge_text = f"PEDESTRIAN WALK PHASE ACTIVE ({crowd_rep.density_level})"
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
        summary_text = (
            f"VEHICLES IN ZONE: {traffic_rep.visible_now} (TOTAL: {traffic_rep.total_unique}) | "
            f"PEDESTRIANS: {crowd_rep.headcount} | "
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.start()
    yield
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
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Only video files (.mp4, .avi, .mov, .mkv) are allowed.")

    content = await file.read()
    meta = pipeline.scenario_manager.save_uploaded_clip(file.filename, content)
    pipeline.select_scenario(meta.id)
    return {"status": "ok", "scenario": {"id": meta.id, "name": meta.name, "category": meta.category}}


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
