"""Tests for Module 9: DecisionEngine (hand-made reports, no models)."""

from app.audio.siren_smoother import AudioEmergencyState
from app.decision.decision_engine import (
    DIRECTION_NOTE, EMERGENCY_MODE, MAINTAIN_ROUTE, MONITOR, NORMAL_MODE, PREPARE_CLEARANCE,
    PRIORITIZE_CLEARANCE, RULE_DESIGNATED, RULE_MOST_CONGESTED, RULE_WHOLE_ROAD, SIGNAL_NOTE,
    DecisionEngine)
from app.lane.lane_analyzer import LaneReport
from app.traffic.traffic_analyzer import TrafficReport


def traffic(level="MEDIUM", density=10.0, visible=10):
    return TrafficReport(timestamp_sec=5.0, total_unique=50, unique_by_class={"car": 50},
                         visible_now=visible, visible_by_class={"car": visible}, load_now=density,
                         density=density, level=level, line_crossings=0, flow_per_minute=0.0)


def lane(name, level, density, visible):
    return LaneReport(name=name, visible_now=visible, visible_by_class={"car": visible},
                      load_now=density, density=density, level=level, unique_vehicles=visible)


def audio(state="NO_SIREN", confidence=0.0, siren_type=None, siren_windows=0):
    return AudioEmergencyState(
        state=state, emergency_active=state == "SIREN_CONFIRMED", window_is_siren=siren_windows > 0,
        window_score=confidence, siren_windows_in_history=siren_windows, history_length=5,
        smoothed_confidence=confidence, siren_type=siren_type,
        event_started_sec=1.0 if state == "SIREN_CONFIRMED" else None,
        event_duration_sec=0.0, timestamp_sec=6.0, events_so_far=0)


LANES = [lane("Lane 1", "HIGH", 7.5, 6), lane("Lane 2", "MEDIUM", 4.0, 4),
         lane("Lane 3", "LOW", 1.0, 1)]
SIREN = audio("SIREN_CONFIRMED", 0.62, "Ambulance", 4)


def test_normal_mode_without_siren():
    decision = DecisionEngine().decide(traffic(), LANES, audio())
    assert decision.mode == NORMAL_MODE and decision.action == MONITOR
    assert decision.headline == "No emergency detected."
    assert decision.recommendation == "Continue normal traffic monitoring."
    assert not decision.emergency_active and decision.target_lane is None


def test_heavy_traffic_without_siren_stays_normal():
    # Brief scenario 2: HIGH TRAFFIC, emergency priority OFF
    decision = DecisionEngine().decide(traffic("HIGH", 20.0, 18), LANES, audio())
    assert decision.mode == NORMAL_MODE
    assert decision.traffic_level == "HIGH"
    assert "heavy" in decision.recommendation.lower()


def test_suspected_siren_is_noted_but_not_an_emergency():
    decision = DecisionEngine().decide(traffic(), LANES, audio("SIREN_SUSPECTED", 0.3, None, 1))
    assert decision.mode == NORMAL_MODE
    assert any("waiting for confirmation" in note for note in decision.notes)


def test_emergency_targets_most_congested_lane_high():
    # Brief scenario 4: congested lane identified, clearance recommendation generated
    decision = DecisionEngine().decide(traffic("HIGH", 18.0), LANES, SIREN)
    assert decision.mode == EMERGENCY_MODE and decision.emergency_active
    assert decision.target_lane == "Lane 1" and decision.target_rule == RULE_MOST_CONGESTED
    assert decision.action == PRIORITIZE_CLEARANCE
    assert decision.headline == "Emergency vehicle detected."
    assert decision.recommendation == ("Lane 1 is highly congested. "
                                       "Prioritize clearance of Lane 1.")


def test_emergency_on_low_traffic_maintains_route():
    # Brief scenario 3: emergency + low congestion -> route relatively clear
    quiet_lanes = [lane("Lane 1", "LOW", 1.0, 1), lane("Lane 2", "LOW", 0.5, 1)]
    decision = DecisionEngine().decide(traffic("LOW", 2.0, 2), quiet_lanes, SIREN)
    assert decision.mode == EMERGENCY_MODE
    assert decision.action == MAINTAIN_ROUTE
    assert decision.recommendation.endswith("Maintain route availability.")


def test_medium_target_prepares_clearance():
    lanes = [lane("Lane 1", "MEDIUM", 4.5, 4), lane("Lane 2", "LOW", 1.0, 1)]
    decision = DecisionEngine().decide(traffic(), lanes, SIREN)
    assert decision.action == PREPARE_CLEARANCE and decision.target_lane == "Lane 1"


def test_designated_lane_is_used_and_other_high_lane_is_mentioned():
    decision = DecisionEngine(emergency_lane="Lane 3").decide(traffic(), LANES, SIREN)
    assert decision.target_lane == "Lane 3" and decision.target_rule == RULE_DESIGNATED
    assert decision.action == MAINTAIN_ROUTE
    assert any("Lane 1 is also HIGH" in note for note in decision.notes)


def test_unknown_designated_lane_falls_back_to_most_congested():
    decision = DecisionEngine(emergency_lane="Bus lane").decide(traffic(), LANES, SIREN)
    assert decision.target_lane == "Lane 1" and decision.target_rule == RULE_MOST_CONGESTED
    assert any("not found" in note for note in decision.notes)


def test_no_lanes_uses_whole_road_level():
    decision = DecisionEngine().decide(traffic("HIGH", 20.0, 15), [], SIREN)
    assert decision.target_lane is None and decision.target_rule == RULE_WHOLE_ROAD
    assert decision.action == PRIORITIZE_CLEARANCE
    assert decision.recommendation == ("The route is highly congested. "
                                       "Prioritize clearance of the route.")


def test_tie_on_density_is_deterministic():
    lanes = [lane("Lane 1", "HIGH", 6.0, 5), lane("Lane 2", "HIGH", 6.0, 5)]
    assert DecisionEngine().decide(traffic(), lanes, SIREN).target_lane == "Lane 1"


def test_emergency_is_explainable_and_honest():
    decision = DecisionEngine().decide(traffic(), LANES, SIREN)
    assert DIRECTION_NOTE in decision.notes and SIGNAL_NOTE in decision.notes
    assert any("0.62" in reason and "Ambulance" in reason for reason in decision.reasons)
    assert any("PRIORITIZE_CLEARANCE" in reason for reason in decision.reasons)
    text = decision.text().lower()
    for forbidden in ["change the signal", "turn green", "switching signal"]:
        assert forbidden not in text


def test_from_config_reads_emergency_lane():
    config = {"decision": {"emergency_lane": "Lane 2"}}
    assert DecisionEngine.from_config(config).emergency_lane == "Lane 2"
    assert DecisionEngine.from_config({}).emergency_lane is None
