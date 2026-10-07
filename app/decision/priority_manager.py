"""
Module 10 - Emergency Priority Logic (state machine).

    manager = PriorityManager.from_config(config)
    status = manager.update(traffic_report, lane_reports, audio_state)
    status.state            # NORMAL / SIREN_DETECTED / EMERGENCY_PRIORITY /
                            # ROUTE_ANALYSIS / RECOMMENDATION / RETURN_TO_NORMAL
    status.priority_active  # True from EMERGENCY_PRIORITY until the event ends
    status.decision         # Module 9 Decision (recommendation, reasons, notes)

    NORMAL ──(siren heard)──► SIREN_DETECTED ──(not confirmed)──► NORMAL
       └────────(siren confirmed by Module 8)────────┐
                                                     ▼
                                           EMERGENCY_PRIORITY   priority switched ON
                                                     ▼
                                           ROUTE_ANALYSIS       average lane densities for a
                                                     ▼          few seconds, choose target lane
                                           RECOMMENDATION       target lane HELD (switch only if
                                                     │          another lane is clearly worse
                                                     │          for several seconds)
                                    (Module 8 ends the event)
                                                     ▼
                                           RETURN_TO_NORMAL     "emergency ended" for a few s
                                                     ▼          (a new siren re-activates)
                                                  NORMAL

Activation threshold + temporal confirmation are applied before this module:
per-window siren threshold (Module 7) and "3 of the last 5 windows" (Module 8).
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace

from app.decision.decision_engine import (
    RULE_DESIGNATED, RULE_MOST_CONGESTED, DecisionEngine)

NORMAL = "NORMAL"
SIREN_DETECTED = "SIREN_DETECTED"
EMERGENCY_PRIORITY = "EMERGENCY_PRIORITY"
ROUTE_ANALYSIS = "ROUTE_ANALYSIS"
RECOMMENDATION = "RECOMMENDATION"
RETURN_TO_NORMAL = "RETURN_TO_NORMAL"
PRIORITY_STATES = {EMERGENCY_PRIORITY, ROUTE_ANALYSIS, RECOMMENDATION}

ANALYSING_ROUTE = "ANALYSING_ROUTE"


@dataclass
class EmergencyEvent:
    """Record of one confirmed emergency (for the dashboard and evaluation)."""
    event_id: int
    confirmed_at_sec: float
    ended_at_sec: float | None = None
    target_lane: str | None = None
    target_rule: str | None = None
    target_switches: list = field(default_factory=list)   # human-readable switch log
    siren_scores: list = field(default_factory=list)      # model scores of SIREN windows only
    siren_types: dict = field(default_factory=lambda: defaultdict(int))
    actions: dict = field(default_factory=lambda: defaultdict(int))  # action -> updates

    @property
    def duration_sec(self):
        return None if self.ended_at_sec is None else self.ended_at_sec - self.confirmed_at_sec

    @property
    def mean_confidence(self):
        return sum(self.siren_scores) / len(self.siren_scores) if self.siren_scores else 0.0

    @property
    def peak_confidence(self):
        return max(self.siren_scores, default=0.0)

    @property
    def siren_type(self):
        return max(self.siren_types, key=self.siren_types.get) if self.siren_types else None


@dataclass
class PriorityStatus:
    state: str
    priority_active: bool
    state_since_sec: float
    decision: object                 # Module 9 Decision, adjusted for the current state
    message: str                     # one-line status for the dashboard
    event: EmergencyEvent | None     # current (or just finished) emergency
    timestamp_sec: float
    transitions: list                # recent (time, from, to, reason)


class PriorityManager:
    def __init__(self, engine=None, route_analysis_seconds=2.0, target_switch_margin=2.0,
                 target_switch_hold_seconds=3.0, return_to_normal_seconds=5.0):
        self.engine = engine or DecisionEngine()
        self.route_analysis_seconds = route_analysis_seconds
        self.target_switch_margin = target_switch_margin
        self.target_switch_hold_seconds = target_switch_hold_seconds
        self.return_to_normal_seconds = return_to_normal_seconds
        self.reset()

    @classmethod
    def from_config(cls, config, engine=None):
        settings = config.get("priority", {})
        return cls(
            engine=engine or DecisionEngine.from_config(config),
            route_analysis_seconds=settings.get("route_analysis_seconds", 2.0),
            target_switch_margin=settings.get("target_switch_margin", 2.0),
            target_switch_hold_seconds=settings.get("target_switch_hold_seconds", 3.0),
            return_to_normal_seconds=settings.get("return_to_normal_seconds", 5.0),
        )

    def reset(self):
        self.state = NORMAL
        self.state_since = 0.0
        self.events = []                     # all emergency events so far
        self.current_event = None
        self.transitions = deque(maxlen=20)
        self._last_audio_time = None
        self._analysis_samples = defaultdict(list)   # lane -> densities during ROUTE_ANALYSIS
        self._switch_candidate = None
        self._switch_since = None

    # ------------------------------------------------------------------ main update

    def update(self, traffic_report, lane_reports, audio_state, now=None):
        """Call whenever new video and/or audio results are available."""
        lane_reports = list(lane_reports or [])
        now = now if now is not None else max(traffic_report.timestamp_sec,
                                              audio_state.timestamp_sec)
        new_audio_window = audio_state.timestamp_sec != self._last_audio_time
        self._last_audio_time = audio_state.timestamp_sec

        if audio_state.emergency_active:
            self._while_siren_confirmed(lane_reports, audio_state, now)
        else:
            self._while_no_confirmed_siren(audio_state, now)

        # Event confidence: real model scores of SIREN windows, each window counted once
        if self.current_event and self.state in PRIORITY_STATES and new_audio_window \
                and audio_state.window_is_siren:
            self.current_event.siren_scores.append(audio_state.window_score)
            if audio_state.siren_type:
                self.current_event.siren_types[audio_state.siren_type] += 1

        decision = self._decision_for_state(traffic_report, lane_reports, audio_state, now)
        if self.current_event and self.state == RECOMMENDATION:
            self.current_event.actions[decision.action] += 1

        return PriorityStatus(
            state=self.state,
            priority_active=self.state in PRIORITY_STATES,
            state_since_sec=self.state_since,
            decision=decision,
            message=self._message(decision, now),
            event=self.current_event or (self.events[-1] if self.events else None),
            timestamp_sec=now,
            transitions=list(self.transitions),
        )

    # ------------------------------------------------------------------ transitions

    def _go(self, new_state, now, reason):
        if new_state != self.state:
            self.transitions.append((round(now, 1), self.state, new_state, reason))
            self.state, self.state_since = new_state, now

    def _while_siren_confirmed(self, lane_reports, audio_state, now):
        if self.state not in PRIORITY_STATES:
            # NORMAL / SIREN_DETECTED / RETURN_TO_NORMAL -> a new emergency event
            self.current_event = EmergencyEvent(event_id=len(self.events) + 1,
                                                confirmed_at_sec=now)
            self.events.append(self.current_event)
            self._go(EMERGENCY_PRIORITY, now,
                     f"siren confirmed ({audio_state.siren_windows_in_history}/"
                     f"{audio_state.history_length} windows)")
            return
        if self.state == EMERGENCY_PRIORITY:
            self._analysis_samples.clear()
            self._go(ROUTE_ANALYSIS, now, "analysing lane conditions")
        if self.state == ROUTE_ANALYSIS:
            for lane in lane_reports:
                self._analysis_samples[lane.name].append(lane.density)
            if now - self.state_since >= self.route_analysis_seconds:
                self._choose_target(lane_reports)
                event = self.current_event
                self._go(RECOMMENDATION, now,
                         f"target: {event.target_lane or 'whole road'} ({event.target_rule})")
            return
        if self.state == RECOMMENDATION:
            self._maybe_switch_target(lane_reports, now)

    def _while_no_confirmed_siren(self, audio_state, now):
        if self.state in PRIORITY_STATES:
            self.current_event.ended_at_sec = now
            self._go(RETURN_TO_NORMAL, now, "siren no longer heard (event ended)")
            return
        if self.state == RETURN_TO_NORMAL:
            if now - self.state_since < self.return_to_normal_seconds:
                return
            self.current_event = None
        heard = audio_state.state == "SIREN_SUSPECTED"
        self._go(SIREN_DETECTED if heard else NORMAL, now,
                 "siren heard, confirming" if heard else "no siren")

    # ------------------------------------------------------------------ target lane

    def _choose_target(self, lane_reports):
        """Option C, but on lane densities AVERAGED over the analysis period."""
        event = self.current_event
        names = [lane.name for lane in lane_reports]
        designated = self.engine.emergency_lane
        if designated and designated in names:
            event.target_lane, event.target_rule = designated, RULE_DESIGNATED
        elif self._analysis_samples:
            averages = {name: sum(values) / len(values)
                        for name, values in self._analysis_samples.items() if values}
            # max() keeps the first lane on ties -> deterministic
            event.target_lane = max(averages, key=averages.get)
            event.target_rule = RULE_MOST_CONGESTED
        else:
            event.target_lane, event.target_rule = None, None   # no lanes -> whole road
        self._switch_candidate = None

    def _maybe_switch_target(self, lane_reports, now):
        """Hysteresis: move the held target only if another lane is clearly worse for a while."""
        event = self.current_event
        if event.target_rule != RULE_MOST_CONGESTED or not lane_reports:
            return   # designated lane (or whole road) is never switched
        by_name = {lane.name: lane for lane in lane_reports}
        held = by_name.get(event.target_lane)
        if held is None:
            return
        rival = max((lane for lane in lane_reports if lane.name != held.name),
                    key=lambda lane: lane.density, default=None)
        if rival is None or rival.density < held.density + self.target_switch_margin:
            self._switch_candidate = None
            return
        if self._switch_candidate != rival.name:
            self._switch_candidate, self._switch_since = rival.name, now
            return
        if now - self._switch_since >= self.target_switch_hold_seconds:
            event.target_switches.append(
                f"{now:.1f}s: {held.name} -> {rival.name} "
                f"(density {rival.density:.1f} vs {held.density:.1f} for "
                f">= {self.target_switch_hold_seconds:g}s)")
            self.transitions.append((round(now, 1), RECOMMENDATION, RECOMMENDATION,
                                     f"target switched {held.name} -> {rival.name}"))
            event.target_lane = rival.name
            self._switch_candidate = None

    # ------------------------------------------------------------------ output

    def _decision_for_state(self, traffic_report, lane_reports, audio_state, now):
        event = self.current_event
        if self.state == RECOMMENDATION:
            return self.engine.decide(traffic_report, lane_reports, audio_state,
                                      target_lane=event.target_lane,
                                      target_rule=f"{event.target_rule} (held)"
                                      if event.target_rule else None)
        decision = self.engine.decide(traffic_report, lane_reports, audio_state)
        if self.state in (EMERGENCY_PRIORITY, ROUTE_ANALYSIS):
            return replace(decision, action=ANALYSING_ROUTE, target_lane=None,
                           target_rule=None,
                           recommendation="Emergency priority activated. "
                                          "Analysing lane conditions to choose the route...")
        if self.state == RETURN_TO_NORMAL:
            return replace(decision, headline="Emergency ended.",
                           recommendation="Siren no longer heard. Returning to normal "
                                          "traffic monitoring.")
        return decision

    def _message(self, decision, now):
        event = self.current_event
        if self.state == NORMAL:
            return f"{decision.headline} {decision.recommendation}"
        if self.state == SIREN_DETECTED:
            return "Possible siren detected - confirming..."
        if self.state in (EMERGENCY_PRIORITY, ROUTE_ANALYSIS):
            return "EMERGENCY PRIORITY ACTIVE - analysing lane conditions..."
        if self.state == RECOMMENDATION:
            return f"EMERGENCY PRIORITY ACTIVE - {decision.recommendation}"
        return (f"Emergency ended after {event.duration_sec:.0f} s. "
                f"Returning to normal monitoring.") if event else "Returning to normal monitoring."
