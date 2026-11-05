"""
Module 11A - Directional Emergency Fusion.

Fuses multimodal audio siren evidence with directional visual camera evidence:
  1. If NO SIREN:
     -> Status: NORMAL
     -> Action: MONITOR
     -> Recommendation: Normal traffic monitoring across all approaches.

  2. If SIREN CONFIRMED:
     -> Case A: Visual emergency vehicle confirmed on a specific approach (e.g. North)
        -> Status: EMERGENCY_ACTIVE
        -> Affected Direction: "North"
        -> Action: PRIORITIZE_CLEARANCE
        -> Recommendation: "Emergency vehicle (Ambulance) spotted on North Approach.
           Prioritize clearance on North Approach immediately."

     -> Case B: No visual emergency vehicle spotted yet on any approach
        -> Status: UNCERTAIN_EMERGENCY
        -> Affected Direction: None ("Unknown")
        -> Action: CAUTION_ALL_APPROACHES
        -> Recommendation: "Emergency siren confirmed, but approach direction is
           unknown. Issue caution alert to all 4 approaches and stand by for clearance."
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.audio.siren_smoother import AudioEmergencyState
from app.traffic.multi_approach_manager import ApproachState
from app.traffic.traffic_analyzer import HIGH, LOW, MEDIUM

STATUS_NORMAL = "NORMAL"
STATUS_EMERGENCY_ACTIVE = "EMERGENCY_ACTIVE"
STATUS_UNCERTAIN_EMERGENCY = "UNCERTAIN_EMERGENCY"

ACTION_MONITOR = "MONITOR"
ACTION_PRIORITIZE_CLEARANCE = "PRIORITIZE_CLEARANCE"
ACTION_CAUTION_ALL = "CAUTION_ALL_APPROACHES"


@dataclass
class DirectionalEmergencyDecision:
    """Multimodal situational assessment of the 4-way intersection."""
    timestamp_sec: float
    system_status: str                     # NORMAL / EMERGENCY_ACTIVE / UNCERTAIN_EMERGENCY
    emergency_confirmed: bool              # True if siren is actively confirmed
    siren_confidence: float                # audio model confidence (0.0 to 1.0)
    siren_type: Optional[str]              # e.g. "Ambulance", "Police", "Fire truck"
    affected_direction_id: Optional[str]   # "north", "south", "east", "west", or None
    affected_direction_name: Optional[str] # "North Approach", etc.
    action: str                            # MONITOR / PRIORITIZE_CLEARANCE / CAUTION_ALL_APPROACHES
    headline: str                          # High-level summary text
    recommendation: str                    # Detailed explainable operational guidance
    reasons: List[str] = field(default_factory=list)
    approach_summaries: Dict[str, dict] = field(default_factory=dict)


class DirectionalFusion:
    """Fuses audio siren detection with multi-approach camera intelligence."""

    def __init__(self, high_congestion_threshold: float = 15.0):
        self.high_congestion_threshold = high_congestion_threshold

    def evaluate(
        self,
        audio_state: AudioEmergencyState,
        approach_states: Dict[str, ApproachState],
    ) -> DirectionalEmergencyDecision:
        """Produce an explainable directional emergency decision."""
        timestamp = audio_state.timestamp_sec if audio_state else 0.0
        emergency_active = audio_state.emergency_active if audio_state else False
        siren_score = audio_state.smoothed_confidence if audio_state else 0.0
        siren_type = audio_state.siren_type if (audio_state and emergency_active) else None

        # Build approach summaries
        summaries = {}
        candidate_emergencies = []
        most_congested_dir = None
        max_density = -1.0

        for dir_id, state in approach_states.items():
            rep = state.traffic_report
            density = rep.density if rep else 0.0
            level = rep.level if rep else LOW
            visible = rep.visible_now if rep else 0
            unique = rep.total_unique if rep else 0

            summaries[dir_id] = {
                "name": state.name,
                "density": density,
                "level": level,
                "visible_now": visible,
                "total_unique": unique,
                "has_emergency_vehicle": state.has_visual_emergency,
                "emergency_label": state.emergency_label,
            }

            if state.has_visual_emergency:
                candidate_emergencies.append((dir_id, state.name, state.emergency_label))

            if density > max_density:
                max_density = density
                most_congested_dir = (dir_id, state.name, density)

        reasons = []

        # Case 1: No active emergency siren
        if not emergency_active:
            reasons.append(f"Audio siren state is {audio_state.state if audio_state else 'INACTIVE'}.")
            if candidate_emergencies:
                names = ", ".join(name for _, name, _ in candidate_emergencies)
                reasons.append(f"Visual candidate detected on {names}, but no sustained siren heard.")
                headline = "Monitoring — Visual Candidate Unverified"
                rec = f"Visual vehicle detected on {names}, but emergency siren is not confirmed. Continuing normal monitoring."
            else:
                headline = "Normal Traffic Monitoring"
                rec = "Traffic is flowing normally across all 4 approaches. No emergency detected."

            return DirectionalEmergencyDecision(
                timestamp_sec=timestamp,
                system_status=STATUS_NORMAL,
                emergency_confirmed=False,
                siren_confidence=siren_score,
                siren_type=None,
                affected_direction_id=None,
                affected_direction_name=None,
                action=ACTION_MONITOR,
                headline=headline,
                recommendation=rec,
                reasons=reasons,
                approach_summaries=summaries,
            )

        # Case 2: Siren CONFIRMED
        reasons.append(f"Sustained emergency siren confirmed by AST model (score {siren_score:.2f}, type: {siren_type or 'Emergency vehicle'}).")

        # Check visual pinpointing
        if len(candidate_emergencies) >= 1:
            target_id, target_name, target_label = candidate_emergencies[0]
            label_text = target_label or siren_type or "Emergency vehicle"
            reasons.append(f"Visual confirmation: {label_text} located on {target_name}.")

            rec = (
                f"PRIORITY ACTION: Emergency {label_text} confirmed on {target_name}. "
                f"Prioritize clearance on {target_name} to ensure rapid transit."
            )

            return DirectionalEmergencyDecision(
                timestamp_sec=timestamp,
                system_status=STATUS_EMERGENCY_ACTIVE,
                emergency_confirmed=True,
                siren_confidence=siren_score,
                siren_type=siren_type or target_label or "Emergency vehicle",
                affected_direction_id=target_id,
                affected_direction_name=target_name,
                action=ACTION_PRIORITIZE_CLEARANCE,
                headline=f"EMERGENCY ACTIVE: {target_name}",
                recommendation=rec,
                reasons=reasons,
                approach_summaries=summaries,
            )

        # Case 3: Siren confirmed, but direction is UNKNOWN visually
        reasons.append("Siren heard, but no emergency vehicle is visible on any of the 4 directional cameras.")
        cong_note = f" (Highest congestion currently on {most_congested_dir[1]} with density {most_congested_dir[2]:.1f})" if most_congested_dir else ""

        rec = (
            f"CAUTION: An active {siren_type or 'emergency'} siren is confirmed by audio sensors, "
            f"but approaching direction is not yet established visually. "
            f"Alert all 4 approaches and maintain vigilance{cong_note}."
        )

        return DirectionalEmergencyDecision(
            timestamp_sec=timestamp,
            system_status=STATUS_UNCERTAIN_EMERGENCY,
            emergency_confirmed=True,
            siren_confidence=siren_score,
            siren_type=siren_type or "Emergency vehicle",
            affected_direction_id=None,
            affected_direction_name="Direction Unknown",
            action=ACTION_CAUTION_ALL,
            headline="SIREN CONFIRMED — DIRECTION UNKNOWN",
            recommendation=rec,
            reasons=reasons,
            approach_summaries=summaries,
        )

