"""
Module 9 - Multimodal Decision Engine.

Combines VIDEO information (overall traffic level, per-lane levels) with AUDIO
information (confirmed emergency, smoothed confidence, siren type) using
simple, explainable rules - no extra ML model:

    engine = DecisionEngine.from_config(config)
    decision = engine.decide(traffic_report, lane_reports, audio_state)
    decision.mode             # "NORMAL" or "EMERGENCY"
    decision.recommendation   # human-readable text
    decision.reasons          # the facts + rules that produced it

Rules
  1. Siren NOT confirmed            -> NORMAL mode, continue monitoring.
  2. Siren confirmed                -> EMERGENCY mode.
  3. Target lane (user choice "C"): the designated emergency lane from config if set,
     otherwise the MOST CONGESTED lane (highest density); whole road if no lanes.
  4. Target level HIGH   -> PRIORITIZE_CLEARANCE
                  MEDIUM -> PREPARE_CLEARANCE
                  LOW    -> MAINTAIN_ROUTE
The system only RECOMMENDS. It does not control traffic signals, and it does
not know where the emergency vehicle is (one microphone gives no direction).
"""

from dataclasses import dataclass, field

from app.traffic.traffic_analyzer import HIGH, LOW, MEDIUM

NORMAL_MODE = "NORMAL"
EMERGENCY_MODE = "EMERGENCY"

MONITOR = "MONITOR"
MAINTAIN_ROUTE = "MAINTAIN_ROUTE"
PREPARE_CLEARANCE = "PREPARE_CLEARANCE"
PRIORITIZE_CLEARANCE = "PRIORITIZE_CLEARANCE"

RULE_DESIGNATED = "designated emergency lane"
RULE_MOST_CONGESTED = "most congested lane"
RULE_WHOLE_ROAD = "whole road (no lane regions configured)"

DIRECTION_NOTE = "Siren direction unknown (single microphone) - target chosen by rule, not by vehicle position."
SIGNAL_NOTE = "Recommendation only - the system does not control traffic signals."


@dataclass
class Decision:
    mode: str                         # NORMAL / EMERGENCY
    action: str                       # MONITOR / MAINTAIN_ROUTE / PREPARE_CLEARANCE / PRIORITIZE_CLEARANCE
    headline: str
    recommendation: str
    reasons: list = field(default_factory=list)   # facts + rules behind the decision
    notes: list = field(default_factory=list)     # caveats shown with the recommendation
    # audio side
    siren_state: str = "NO_SIREN"
    siren_confidence: float = 0.0
    siren_type: str | None = None
    # video side
    traffic_level: str = LOW
    traffic_density: float = 0.0
    vehicles_in_view: int = 0
    total_vehicles: int = 0
    # target
    target_lane: str | None = None
    target_rule: str | None = None
    target_level: str | None = None
    target_density: float | None = None
    target_vehicles: int | None = None
    timestamp_sec: float = 0.0

    @property
    def emergency_active(self):
        return self.mode == EMERGENCY_MODE

    def text(self):
        lines = [self.headline, self.recommendation]
        lines += [f"  - {reason}" for reason in self.reasons]
        lines += [f"  ({note})" for note in self.notes]
        return "\n".join(lines)


class DecisionEngine:
    def __init__(self, emergency_lane=None):
        self.emergency_lane = emergency_lane

    @classmethod
    def from_config(cls, config):
        return cls(emergency_lane=config.get("decision", {}).get("emergency_lane"))

    def decide(self, traffic_report, lane_reports, audio_state, target_lane=None,
               target_rule=None):
        """One snapshot decision from the latest video and audio results.

        target_lane / target_rule (optional): a lane already chosen and held by the
        priority state machine (Module 10). Without them the target is chosen fresh.
        """
        lane_reports = list(lane_reports or [])
        common = dict(
            siren_state=audio_state.state,
            siren_confidence=audio_state.smoothed_confidence,
            siren_type=audio_state.siren_type,
            traffic_level=traffic_report.level,
            traffic_density=traffic_report.density,
            vehicles_in_view=traffic_report.visible_now,
            total_vehicles=traffic_report.total_unique,
            timestamp_sec=max(traffic_report.timestamp_sec, audio_state.timestamp_sec),
        )
        traffic_fact = (f"Overall traffic {traffic_report.level} "
                        f"(density {traffic_report.density:.1f}, {traffic_report.visible_now} "
                        f"vehicles in view)")

        # ---------------- Rule 1: no confirmed siren -> NORMAL
        if not audio_state.emergency_active:
            reasons = [f"Siren not confirmed ({audio_state.state}, "
                       f"{audio_state.siren_windows_in_history}/{audio_state.history_length} "
                       f"recent windows siren)", traffic_fact]
            notes = []
            if audio_state.state == "SIREN_SUSPECTED":
                notes.append("Possible siren heard - waiting for confirmation.")
            recommendation = "Continue normal traffic monitoring."
            if traffic_report.level == HIGH:
                recommendation = "Traffic is heavy. Continue normal traffic monitoring."
            return Decision(mode=NORMAL_MODE, action=MONITOR,
                            headline="No emergency detected.",
                            recommendation=recommendation, reasons=reasons, notes=notes,
                            **common)

        # ---------------- Rules 2 + 3: confirmed siren -> EMERGENCY, choose target
        target, rule, notes = self._choose_target(lane_reports)
        if target_lane is not None:
            # The priority state machine (Module 10) holds a target chosen earlier
            held = next((lane for lane in lane_reports if lane.name == target_lane), None)
            if held is not None:
                target, rule = held, target_rule or rule
        if target is not None:
            level, density, vehicles, place = (target.level, target.density,
                                               target.visible_now, target.name)
        else:
            level, density, vehicles, place = (traffic_report.level, traffic_report.density,
                                               traffic_report.visible_now, "The route")

        # ---------------- Rule 4: target level -> action
        if level == HIGH:
            action = PRIORITIZE_CLEARANCE
            condition = f"{place} is highly congested."
            advice = f"Prioritize clearance of {place if target else 'the route'}."
        elif level == MEDIUM:
            action = PREPARE_CLEARANCE
            condition = f"{place} has moderate traffic."
            advice = f"Prepare to clear {place if target else 'the route'}."
        else:
            action = MAINTAIN_ROUTE
            condition = f"{place} currently has low congestion."
            advice = "Maintain route availability."

        siren_text = f"Emergency siren confirmed (smoothed confidence {audio_state.smoothed_confidence:.2f}"
        siren_text += f", likely {audio_state.siren_type})" if audio_state.siren_type else ")"
        reasons = [siren_text, traffic_fact,
                   f"Target: {place} chosen as {rule} "
                   f"({vehicles} vehicles, density {density:.1f}, level {level})",
                   f"Rule: target level {level} -> {action}"]
        notes += [DIRECTION_NOTE, SIGNAL_NOTE]
        return Decision(mode=EMERGENCY_MODE, action=action,
                        headline="Emergency vehicle detected.",
                        recommendation=f"{condition} {advice}",
                        reasons=reasons, notes=notes,
                        target_lane=target.name if target else None, target_rule=rule,
                        target_level=level, target_density=density, target_vehicles=vehicles,
                        **common)

    def _choose_target(self, lane_reports):
        """Option C: designated lane if configured and present, else most congested lane."""
        notes = []
        if not lane_reports:
            return None, RULE_WHOLE_ROAD, notes
        by_name = {lane.name: lane for lane in lane_reports}
        if self.emergency_lane:
            if self.emergency_lane in by_name:
                target = by_name[self.emergency_lane]
                worst = max(lane_reports, key=lambda lane: lane.density)
                if worst.name != target.name and worst.level == HIGH and target.level != HIGH:
                    notes.append(f"{worst.name} is also HIGH (density {worst.density:.1f}).")
                return target, RULE_DESIGNATED, notes
            notes.append(f"Designated lane '{self.emergency_lane}' not found in this camera's "
                         f"lanes - using the most congested lane instead.")
        # max() keeps the first lane on ties -> deterministic
        target = max(lane_reports, key=lambda lane: lane.density)
        return target, RULE_MOST_CONGESTED, notes
