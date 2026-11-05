"""Unit tests for Module 11A Directional Emergency Fusion."""

import pytest

from app.audio.siren_smoother import AudioEmergencyState
from app.decision.directional_fusion import (
    ACTION_CAUTION_ALL, ACTION_MONITOR, ACTION_PRIORITIZE_CLEARANCE,
    STATUS_EMERGENCY_ACTIVE, STATUS_NORMAL, STATUS_UNCERTAIN_EMERGENCY,
    DirectionalFusion)
from app.traffic.multi_approach_manager import ApproachState
from app.traffic.traffic_analyzer import HIGH, LOW, MEDIUM, TrafficReport


def make_report(level="LOW", density=2.0, visible=2, unique=5):
    return TrafficReport(
        timestamp_sec=1.0,
        total_unique=unique,
        unique_by_class={"car": unique},
        visible_now=visible,
        visible_by_class={"car": visible},
        load_now=density,
        density=density,
        level=level,
        line_crossings=0,
        flow_per_minute=0.0,
    )


def make_approach_state(direction_id, name, level="LOW", density=2.0, has_emergency=False, label=None):
    return ApproachState(
        direction_id=direction_id,
        name=name,
        frame=None,
        annotated_image=None,
        detections=[],
        traffic_report=make_report(level=level, density=density),
        has_visual_emergency=has_emergency,
        emergency_label=label,
    )


def test_normal_monitoring_when_no_siren():
    fusion = DirectionalFusion()
    audio = AudioEmergencyState(
        state="NO_SIREN", emergency_active=False, window_is_siren=False,
        window_score=0.01, siren_windows_in_history=0, history_length=5,
        smoothed_confidence=0.01, siren_type=None, event_started_sec=None,
        event_duration_sec=0.0, timestamp_sec=5.0, events_so_far=0,
    )
    approaches = {
        "north": make_approach_state("north", "North Approach", "LOW", 2.0),
        "south": make_approach_state("south", "South Approach", "MEDIUM", 6.0),
        "east": make_approach_state("east", "East Approach", "LOW", 1.5),
        "west": make_approach_state("west", "West Approach", "HIGH", 16.0),
    }

    decision = fusion.evaluate(audio, approaches)

    assert decision.system_status == STATUS_NORMAL
    assert decision.action == ACTION_MONITOR
    assert decision.affected_direction_id is None
    assert not decision.emergency_confirmed
    assert "Normal Traffic Monitoring" in decision.headline


def test_uncertain_emergency_when_siren_active_but_direction_unknown():
    fusion = DirectionalFusion()
    audio = AudioEmergencyState(
        state="SIREN_CONFIRMED", emergency_active=True, window_is_siren=True,
        window_score=0.85, siren_windows_in_history=4, history_length=5,
        smoothed_confidence=0.72, siren_type="Ambulance", event_started_sec=10.0,
        event_duration_sec=4.0, timestamp_sec=14.0, events_so_far=1,
    )
    approaches = {
        "north": make_approach_state("north", "North Approach", "LOW", 2.0, has_emergency=False),
        "south": make_approach_state("south", "South Approach", "MEDIUM", 7.0, has_emergency=False),
        "east": make_approach_state("east", "East Approach", "LOW", 1.0, has_emergency=False),
        "west": make_approach_state("west", "West Approach", "HIGH", 18.0, has_emergency=False),
    }

    decision = fusion.evaluate(audio, approaches)

    assert decision.system_status == STATUS_UNCERTAIN_EMERGENCY
    assert decision.action == ACTION_CAUTION_ALL
    assert decision.affected_direction_name == "Direction Unknown"
    assert decision.emergency_confirmed
    assert "DIRECTION UNKNOWN" in decision.headline


def test_priority_clearance_when_siren_and_visual_direction_confirmed():
    fusion = DirectionalFusion()
    audio = AudioEmergencyState(
        state="SIREN_CONFIRMED", emergency_active=True, window_is_siren=True,
        window_score=0.92, siren_windows_in_history=5, history_length=5,
        smoothed_confidence=0.88, siren_type="Ambulance", event_started_sec=5.0,
        event_duration_sec=5.0, timestamp_sec=10.0, events_so_far=1,
    )
    approaches = {
        "north": make_approach_state("north", "North Approach", "HIGH", 16.0, has_emergency=True, label="Ambulance"),
        "south": make_approach_state("south", "South Approach", "MEDIUM", 6.0),
        "east": make_approach_state("east", "East Approach", "LOW", 2.0),
        "west": make_approach_state("west", "West Approach", "LOW", 1.0),
    }

    decision = fusion.evaluate(audio, approaches)

    assert decision.system_status == STATUS_EMERGENCY_ACTIVE
    assert decision.action == ACTION_PRIORITIZE_CLEARANCE
    assert decision.affected_direction_id == "north"
    assert decision.affected_direction_name == "North Approach"
    assert "North Approach" in decision.recommendation

