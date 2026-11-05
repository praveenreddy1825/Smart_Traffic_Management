"""
Tests for End-to-End Scenario Logic and Multimodal Decision Integrity.

Validates the operational scenarios across the unified platform:
  1. Baseline Normal Traffic (Routine Signal Cycling)
  2. Congested Vehicle Queue (Green Phase Extension)
  3. Pedestrian Crosswalk & Crowd Gathering (Walk Phase Hold)
  4. Intermittent / Masked Siren (Acoustic Gap Bridging)
  5. Transient Spike Rejection (False Positive Filtering)
  6. Emergency Vehicle & Siren Preemption (Instant Clearance Corridor)
"""

import json
from pathlib import Path
import pytest

from app.analysis.crowd_analyzer import (
    CROWD_CROWDED,
    CROWD_MODERATE,
    CROWD_SPARSE,
    CROWD_SURGE,
    CrowdAnalyzer,
    CrowdReport,
)
from app.audio.siren_classifier import SirenPrediction
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
    UnifiedDecisionEngine,
)
from app.detection.vehicle_detector import Detection
from app.traffic.traffic_analyzer import HIGH, LOW, MEDIUM, TrafficReport
from app.utils.config import load_config, resolve_path
from app.video.scenario_manager import ScenarioManager


def make_traffic_report(t: float, level="HIGH", density=16.0, visible=12) -> TrafficReport:
    return TrafficReport(
        timestamp_sec=t,
        total_unique=visible * 2,
        unique_by_class={"car": visible},
        visible_now=visible,
        visible_by_class={"car": visible},
        load_now=density,
        density=density,
        level=level,
        line_crossings=5,
        flow_per_minute=25.0,
    )


def make_crowd_report(t: float, headcount: int = 12, density: str = CROWD_CROWDED, crosswalk: bool = True) -> CrowdReport:
    return CrowdReport(
        timestamp_sec=t,
        headcount=headcount,
        cumulative_pedestrians=headcount + 5,
        density_level=density,
        cluster_count=2 if headcount >= 10 else 0,
        crosswalk_occupied=crosswalk,
        safety_alert="Pedestrian Crossing Alert" if crosswalk else None,
        recommendation="Trigger walk phase",
    )


def make_audio_state(t: float, state: str, score: float = 0.0) -> AudioEmergencyState:
    is_active = state == "SIREN_CONFIRMED"
    return AudioEmergencyState(
        state=state,
        emergency_active=is_active,
        window_is_siren=score >= 0.15,
        window_score=score,
        siren_windows_in_history=3 if is_active else (1 if state == "SIREN_SUSPECTED" else 0),
        history_length=5,
        smoothed_confidence=score,
        siren_type="Ambulance" if score >= 0.15 else None,
        event_started_sec=t if is_active else None,
        event_duration_sec=2.0 if is_active else 0.0,
        timestamp_sec=t,
        events_so_far=1 if is_active else 0,
    )


def make_pred(is_siren: bool, score: float, t: float) -> SirenPrediction:
    return SirenPrediction(
        is_siren=is_siren,
        siren_score=score,
        siren_type="Ambulance" if is_siren else None,
        type_scores={"Ambulance": score} if is_siren else {},
        top_labels=[("Siren" if is_siren else "Noise", score)],
        inference_ms=15.0,
        start_time_sec=t,
    )


def test_scenario_files_exist():
    """Verify that all four standard audio scenario audio files exist in data/audio/scenarios/."""
    scenarios_json_path = resolve_path("data/audio/scenarios/scenarios.json")
    assert scenarios_json_path.is_file(), "scenarios.json metadata missing"

    with open(scenarios_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for item in meta:
        wav_file = resolve_path(f"data/audio/scenarios/{item['file']}")
        assert wav_file.is_file(), f"Scenario WAV missing: {item['file']}"


def test_scenario_manager_catalog():
    """Verify ScenarioManager catalogs videos across vehicles, crowd, and emergency folders."""
    sm = ScenarioManager(resolve_path("data/videos"))
    scenarios = sm.scan_scenarios()
    assert len(scenarios) >= 3, "Expected at least 3 curated presentation scenarios"

    categories = {s.category for s in scenarios}
    assert "vehicles" in categories
    assert "crowd" in categories
    assert "emergency" in categories


def test_scenario_1_baseline_traffic_no_siren():
    """Scenario 1: Baseline traffic with no siren -> System maintains routine signal cycling."""
    engine = UnifiedDecisionEngine()

    tr = make_traffic_report(1.0, level="LOW", density=3.0, visible=3)
    cr = make_crowd_report(1.0, headcount=1, density=CROWD_SPARSE, crosswalk=False)
    audio = make_audio_state(1.0, "NO_SIREN", score=0.01)

    decision = engine.evaluate(tr, cr, audio, active_mode="auto")
    assert decision.system_status == STATUS_NORMAL
    assert decision.action == "ROUTINE CYCLING"
    assert not decision.has_emergency_vehicle


def test_scenario_2_vehicle_congestion_extension():
    """Scenario 2: Heavy vehicle queue -> System triggers Green Extension to clear queue."""
    engine = UnifiedDecisionEngine()

    tr = make_traffic_report(10.0, level="HIGH", density=18.0, visible=16)
    cr = make_crowd_report(10.0, headcount=2, density=CROWD_SPARSE, crosswalk=False)
    audio = make_audio_state(10.0, "NO_SIREN", score=0.02)

    decision = engine.evaluate(tr, cr, audio, active_mode="vehicle")
    assert decision.system_status == STATUS_CONGESTION_PRIORITY
    assert decision.signal_phase == PHASE_GREEN
    assert decision.recommended_duration_sec >= 25
    assert "EXTENDING GREEN" in decision.headline


def test_scenario_3_crowd_pedestrian_walk_phase():
    """Scenario 3: Pedestrians at crosswalk -> System holds vehicle red and activates walk phase."""
    engine = UnifiedDecisionEngine()

    tr = make_traffic_report(15.0, level="MEDIUM", density=7.0, visible=6)
    cr = make_crowd_report(15.0, headcount=14, density=CROWD_CROWDED, crosswalk=True)
    audio = make_audio_state(15.0, "NO_SIREN", score=0.01)

    decision = engine.evaluate(tr, cr, audio, active_mode="crowd")
    assert decision.system_status == STATUS_PEDESTRIAN_CROSSING
    assert decision.signal_phase == PHASE_PEDESTRIAN_WALK
    assert "PEDESTRIAN WALK PHASE" in decision.headline
    assert "HOLD VEHICLE RED" in decision.action


def test_scenario_4_intermittent_siren_smoothing_bridging():
    """Scenario 4: Intermittent siren with 3-second gap -> Smoothing bridges the gap."""
    smoother = SirenSmoother(history_windows=5, windows_to_activate=3, quiet_windows_to_end=6)

    # 3 consecutive sirens activate emergency
    for t in [1.0, 2.0, 3.0]:
        state = smoother.update(make_pred(True, 0.75, t))
    assert state.emergency_active
    assert state.state == "SIREN_CONFIRMED"

    # 2 non-sirens (gap)
    for t in [4.0, 5.0]:
        state = smoother.update(make_pred(False, 0.02, t))
    assert state.emergency_active
    assert state.state == "SIREN_CONFIRMED"

    # Siren resumes
    state = smoother.update(make_pred(True, 0.80, 6.0))
    assert state.emergency_active
    assert smoother.events_so_far == 1


def test_scenario_5_transient_burst_rejection():
    """Scenario 5: 1-window transient spike (car horn) -> Must NOT trigger confirmed emergency."""
    smoother = SirenSmoother(history_windows=5, windows_to_activate=3, quiet_windows_to_end=6)

    s1 = smoother.update(make_pred(True, 0.60, 1.0))
    assert s1.state == "SIREN_SUSPECTED"
    assert not s1.emergency_active

    s2 = smoother.update(make_pred(False, 0.01, 2.0))
    assert not s2.emergency_active
    assert smoother.events_so_far == 0


def test_scenario_6_emergency_vehicle_and_siren_preemption():
    """Scenario 6: Acoustic siren confirmed + ambulance spotted -> Immediate Green Corridor."""
    engine = UnifiedDecisionEngine()

    tr = make_traffic_report(20.0, level="HIGH", density=15.0, visible=10)
    cr = make_crowd_report(20.0, headcount=5, density=CROWD_MODERATE, crosswalk=True)
    audio_active = make_audio_state(20.0, "SIREN_CONFIRMED", score=0.92)

    decision = engine.evaluate(
        tr, cr, audio_active,
        active_mode="auto",
        has_visual_emergency=True,
        emergency_label="Ambulance",
    )
    assert decision.system_status == STATUS_EMERGENCY_PREEMPTION
    assert decision.signal_phase == PHASE_EMERGENCY_CLEARANCE
    assert decision.action == "CLEAR CORRIDOR IMMEDIATELY"
    assert "AMBULANCE" in decision.headline
    assert decision.has_emergency_vehicle
