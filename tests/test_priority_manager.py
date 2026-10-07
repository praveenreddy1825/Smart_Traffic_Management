"""Tests for Module 10: PriorityManager state machine (hand-made inputs, no models).

Time advances 1 s per update (one audio window per second).
"""

import pytest

from app.audio.siren_smoother import AudioEmergencyState
from app.decision.decision_engine import (
    PRIORITIZE_CLEARANCE, RULE_DESIGNATED, RULE_MOST_CONGESTED, DecisionEngine)
from app.decision.priority_manager import (
    ANALYSING_ROUTE, EMERGENCY_PRIORITY, NORMAL, RECOMMENDATION, RETURN_TO_NORMAL,
    ROUTE_ANALYSIS, SIREN_DETECTED, PriorityManager)
from app.lane.lane_analyzer import LaneReport
from app.traffic.traffic_analyzer import TrafficReport


def traffic(t, level="HIGH", density=16.0):
    return TrafficReport(timestamp_sec=t, total_unique=40, unique_by_class={"car": 40},
                         visible_now=12, visible_by_class={"car": 12}, load_now=density,
                         density=density, level=level, line_crossings=0, flow_per_minute=0.0)


def lanes(d1, d2, d3=1.0):
    level = lambda d: "HIGH" if d >= 6 else ("MEDIUM" if d >= 3 else "LOW")  # noqa: E731
    return [LaneReport(f"Lane {i}", int(d), {"car": int(d)}, d, d, level(d), int(d))
            for i, d in enumerate([d1, d2, d3], 1)]


def audio(t, state, score=0.0):
    active = state == "SIREN_CONFIRMED"
    return AudioEmergencyState(
        state=state, emergency_active=active, window_is_siren=score >= 0.15, window_score=score,
        siren_windows_in_history=3 if active else (1 if state == "SIREN_SUSPECTED" else 0),
        history_length=5, smoothed_confidence=score, siren_type="Ambulance" if score >= 0.15 else None,
        event_started_sec=None, event_duration_sec=0.0, timestamp_sec=t, events_so_far=0)


def manager(**settings):
    defaults = dict(route_analysis_seconds=2.0, target_switch_margin=2.0,
                    target_switch_hold_seconds=3.0, return_to_normal_seconds=5.0)
    defaults.update(settings)
    engine = DecisionEngine(emergency_lane=defaults.pop("emergency_lane", None))
    return PriorityManager(engine=engine, **defaults)


# audio timeline: 0-2 nothing, 3 suspected, 4-14 confirmed (scores 0.6), 15+ nothing
def audio_at(t):
    if t < 3:
        return audio(t, "NO_SIREN")
    if t == 3:
        return audio(t, "SIREN_SUSPECTED", 0.5)
    if t <= 14:
        return audio(t, "SIREN_CONFIRMED", 0.6)
    return audio(t, "NO_SIREN")


def run(pm, seconds, lane_fn=lambda t: lanes(7.0, 4.0), audio_fn=audio_at):
    return [pm.update(traffic(t), lane_fn(t), audio_fn(t)) for t in range(seconds)]


def test_full_state_sequence():
    states = [s.state for s in run(manager(), 25)]
    assert states[0:3] == [NORMAL] * 3
    assert states[3] == SIREN_DETECTED
    assert states[4] == EMERGENCY_PRIORITY
    assert states[5:7] == [ROUTE_ANALYSIS] * 2            # 2 s of route analysis
    assert states[7:15] == [RECOMMENDATION] * 8
    assert states[15:20] == [RETURN_TO_NORMAL] * 5        # 5 s "emergency ended"
    assert states[20:] == [NORMAL] * 5


def test_priority_flag_and_recommendation():
    statuses = run(manager(), 25)
    assert [s.priority_active for s in statuses].count(True) == 11   # t = 4..14
    assert statuses[5].decision.action == ANALYSING_ROUTE
    recommendation = statuses[8].decision
    assert recommendation.action == PRIORITIZE_CLEARANCE
    assert recommendation.target_lane == "Lane 1"
    assert "(held)" in recommendation.target_rule
    assert statuses[16].decision.headline == "Emergency ended."
    assert "Returning to normal" in statuses[16].message


def test_false_alarm_goes_back_to_normal():
    def audio_fn(t):
        return audio(t, "SIREN_SUSPECTED", 0.4) if t in (3, 4) else audio(t, "NO_SIREN")
    states = [s.state for s in run(manager(), 8, audio_fn=audio_fn)]
    assert states == [NORMAL] * 3 + [SIREN_DETECTED] * 2 + [NORMAL] * 3


def test_route_analysis_uses_average_not_last_snapshot():
    # Lane 2 is worse on average during analysis, Lane 1 only spikes at the last moment
    samples = {4: (2.0, 8.0), 5: (2.0, 8.0), 6: (9.0, 8.0)}
    statuses = run(manager(), 9, lane_fn=lambda t: lanes(*samples.get(t, (2.0, 8.0))))
    assert statuses[7].event.target_lane == "Lane 2"
    assert statuses[7].event.target_rule == RULE_MOST_CONGESTED


def test_target_held_during_small_fluctuations():
    # Module 9 problem: densities 4.5 vs 4.4 swapped the target. Here they keep swapping.
    def lane_fn(t):
        return lanes(4.5, 4.4) if t % 2 else lanes(4.4, 4.5)
    statuses = run(manager(), 15, lane_fn=lane_fn)
    targets = {s.decision.target_lane for s in statuses if s.state == RECOMMENDATION}
    assert len(targets) == 1
    assert statuses[14].event.target_switches == []


def test_target_switches_when_other_lane_clearly_worse_for_long_enough():
    def lane_fn(t):
        return lanes(7.0, 4.0) if t < 9 else lanes(4.0, 9.0)   # from t = 9 Lane 2 is +5
    statuses = run(manager(), 15, lane_fn=lane_fn)
    assert statuses[10].decision.target_lane == "Lane 1"   # waiting (hold 3 s)
    assert statuses[12].decision.target_lane == "Lane 2"   # switched at t = 12
    assert len(statuses[12].event.target_switches) == 1
    assert "Lane 1 -> Lane 2" in statuses[12].event.target_switches[0]


def test_short_spike_does_not_switch():
    def lane_fn(t):
        return lanes(4.0, 9.0) if t in (9, 10) else lanes(7.0, 4.0)
    statuses = run(manager(), 15, lane_fn=lane_fn)
    assert all(s.decision.target_lane == "Lane 1" for s in statuses if s.state == RECOMMENDATION)


def test_designated_lane_is_never_switched():
    statuses = run(manager(emergency_lane="Lane 3"), 15, lane_fn=lambda t: lanes(9.0, 9.0, 0.5))
    recommended = [s.decision for s in statuses if s.state == RECOMMENDATION]
    assert all(d.target_lane == "Lane 3" for d in recommended)
    assert statuses[10].event.target_rule == RULE_DESIGNATED


def test_new_siren_during_return_starts_new_event():
    def audio_fn(t):
        if 4 <= t <= 8 or 11 <= t <= 16:
            return audio(t, "SIREN_CONFIRMED", 0.6)
        return audio(t, "NO_SIREN")
    statuses = run(manager(), 18, audio_fn=audio_fn)
    assert statuses[9].state == RETURN_TO_NORMAL
    assert statuses[11].state == EMERGENCY_PRIORITY
    assert statuses[11].event.event_id == 2


def test_event_confidence_uses_siren_windows_once_each():
    pm = manager()
    for t in range(4):
        pm.update(traffic(t), lanes(7, 4), audio_at(t))
    scores = {4: 0.6, 5: 0.2, 6: 0.05, 7: 0.9}   # 0.05 is a non-siren window -> ignored
    for t, score in scores.items():
        state = audio(t, "SIREN_CONFIRMED", score)
        for _ in range(3):                         # called 3x per audio window (e.g. per frame)
            status = pm.update(traffic(t), lanes(7, 4), state, now=t)
    event = status.event
    assert event.siren_scores == [0.6, 0.2, 0.9]
    assert event.mean_confidence == pytest.approx((0.6 + 0.2 + 0.9) / 3)
    assert event.peak_confidence == pytest.approx(0.9)
    assert event.siren_type == "Ambulance"


def test_event_log_records_times_and_actions():
    statuses = run(manager(), 22)
    event = statuses[-1].event
    assert event.confirmed_at_sec == 4 and event.ended_at_sec == 15
    assert event.duration_sec == 11
    assert event.actions[PRIORITIZE_CLEARANCE] == 8
    reasons = [t[3] for t in statuses[-1].transitions]
    assert any("siren confirmed" in r for r in reasons)
    assert any("event ended" in r for r in reasons)


def test_no_lanes_uses_whole_road():
    statuses = run(manager(), 12, lane_fn=lambda t: [])
    decision = statuses[9].decision
    assert statuses[9].state == RECOMMENDATION
    assert decision.target_lane is None and decision.action == PRIORITIZE_CLEARANCE
