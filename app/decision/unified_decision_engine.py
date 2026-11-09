"""
Unified Multimodal Decision Engine.

Fuses Vehicle Traffic Analytics, Crowd & Pedestrian Safety States, and
AST Audio Siren Evidence to make intelligent traffic signal dispatch decisions.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from app.analysis.crowd_analyzer import (
    CROWD_CROWDED,
    CROWD_MODERATE,
    CROWD_SPARSE,
    CROWD_SURGE,
    CrowdReport,
)
from app.audio.siren_smoother import AudioEmergencyState
from app.traffic.traffic_analyzer import HIGH, LOW, MEDIUM, TrafficReport

STATUS_NORMAL = "NORMAL"
STATUS_EMERGENCY_PREEMPTION = "EMERGENCY_PREEMPTION"
STATUS_PEDESTRIAN_CROSSING = "PEDESTRIAN_CROSSING"
STATUS_CONGESTION_PRIORITY = "CONGESTION_PRIORITY"

PHASE_GREEN = "GREEN"
PHASE_YELLOW = "YELLOW"
PHASE_RED = "RED"
PHASE_EMERGENCY_CLEARANCE = "EMERGENCY_CLEARANCE"
PHASE_PEDESTRIAN_WALK = "PEDESTRIAN_WALK"


@dataclass
class UnifiedDecision:
    """Multimodal verdict combining vehicles, pedestrians/crowds, and audio sirens."""
    timestamp_sec: float
    system_status: str                     # NORMAL / EMERGENCY_PREEMPTION / PEDESTRIAN_CROSSING / CONGESTION_PRIORITY
    signal_phase: str                      # GREEN / RED / EMERGENCY_CLEARANCE / PEDESTRIAN_WALK
    recommended_duration_sec: int          # Recommended duration for the signal phase
    headline: str                          # Bold banner headline
    action: str                            # Operational action
    recommendation: str                    # Detailed explainable guidance
    reasons: List[str] = field(default_factory=list)
    has_emergency_vehicle: bool = False
    siren_confidence: float = 0.0


class UnifiedDecisionEngine:
    """Determines real-time traffic signal timing and preemption phases."""

    def evaluate(
        self,
        vehicle_report: Optional[TrafficReport],
        crowd_report: Optional[CrowdReport],
        audio_state: Optional[AudioEmergencyState],
        active_mode: str = "auto",
        has_visual_emergency: bool = False,
        emergency_label: Optional[str] = None,
    ) -> UnifiedDecision:
        timestamp = vehicle_report.timestamp_sec if vehicle_report else (crowd_report.timestamp_sec if crowd_report else 0.0)

        siren_active = audio_state.emergency_active if audio_state else False
        siren_conf = audio_state.smoothed_confidence if audio_state else 0.0
        siren_type = audio_state.siren_type if audio_state else None

        # Priority 1: Emergency Vehicle Preemption (Siren sound or visual emergency vehicle)
        if siren_active or has_visual_emergency:
            label = emergency_label or siren_type or "Ambulance"
            reasons = []
            if siren_active:
                reasons.append(f"Acoustic emergency siren confirmed by AST model ({siren_conf*100:.0f}% confidence)")
            if has_visual_emergency:
                reasons.append(f"Visual {label} identified in approach camera")

            return UnifiedDecision(
                timestamp_sec=timestamp,
                system_status=STATUS_EMERGENCY_PREEMPTION,
                signal_phase=PHASE_EMERGENCY_CLEARANCE,
                recommended_duration_sec=35,
                headline=f"🚨 EMERGENCY PREEMPTION: {label.upper()} APPROACHING",
                action="CLEAR CORRIDOR IMMEDIATELY",
                recommendation=f"Preempt all conflicting traffic lights. Provide instant green corridor for approaching {label}.",
                reasons=reasons,
                has_emergency_vehicle=True,
                siren_confidence=siren_conf,
            )

        # Priority 2: Crowd / Pedestrian Crossing Priority
        # Active when crowd mode is selected, or in auto mode when high foot traffic is present
        headcount = crowd_report.headcount if crowd_report else 0
        density = crowd_report.density_level if crowd_report else CROWD_SPARSE

        if active_mode in ("crowd", "auto") and (density in (CROWD_CROWDED, CROWD_SURGE) or (active_mode == "crowd" and headcount >= 3)):
            is_crowd = (active_mode == "crowd") or (density == CROWD_SURGE)
            est_total = crowd_report.estimated_total_crowd if crowd_report else headcount
            reasons = [
                f"Crowd headcount in view: {headcount} ({density} density, ~{est_total} est. gathering)" if is_crowd else f"Pedestrian headcount in view: {headcount} ({density} density)",
                f"Crowd monitored zone occupied: {crowd_report.crosswalk_occupied if crowd_report else False}" if is_crowd else f"Crosswalk occupied: {crowd_report.crosswalk_occupied if crowd_report else False}",
            ]
            if crowd_report and crowd_report.cluster_count > 0:
                reasons.append(f"Detected {crowd_report.cluster_count} dense gathering cluster(s)")

            walk_duration = 25 if density == CROWD_SURGE else (15 if density == CROWD_CROWDED else 12)
            if density == CROWD_SURGE:
                headline = f"🚨 CRITICAL CROWD SURGE CLEARANCE ({density})"
                action = f"HOLD VEHICULAR RED (CROWD CLEARANCE {walk_duration}s)"
                rec = f"Hold vehicle traffic on red for {walk_duration}s to guarantee public safety for ~{est_total} gathering persons."
            elif is_crowd:
                headline = f"👥 CROWD SAFETY CLEARANCE ACTIVE ({density})"
                action = f"HOLD VEHICULAR RED (CROWD CLEARANCE {walk_duration}s)"
                rec = f"Hold vehicle traffic on red for {walk_duration}s to manage safe crowd flow for {headcount} detected persons."
            else:
                headline = f"👥 PEDESTRIAN WALK PHASE ACTIVE ({density})"
                action = "HOLD VEHICLE RED / ACTIVATE WALK SIGNAL"
                rec = f"Hold vehicle traffic on red for {walk_duration}s to guarantee safe crossing for {headcount} pedestrians."

            return UnifiedDecision(
                timestamp_sec=timestamp,
                system_status=STATUS_PEDESTRIAN_CROSSING,
                signal_phase=PHASE_PEDESTRIAN_WALK,
                recommended_duration_sec=walk_duration,
                headline=headline,
                action=action,
                recommendation=rec,
                reasons=reasons,
                siren_confidence=siren_conf,
            )

        # Priority 3: Heavy Vehicle Traffic Congestion
        veh_level = vehicle_report.level if vehicle_report else LOW
        visible_vecs = vehicle_report.visible_now if vehicle_report else 0

        if active_mode in ("vehicle", "auto") and (veh_level == HIGH or visible_vecs >= 10):
            reasons = [
                f"Traffic load level is HIGH ({visible_vecs} vehicles in view)",
                f"Queue buildup detected on approach",
            ]
            green_duration = 30 if visible_vecs > 15 else 22
            return UnifiedDecision(
                timestamp_sec=timestamp,
                system_status=STATUS_CONGESTION_PRIORITY,
                signal_phase=PHASE_GREEN,
                recommended_duration_sec=green_duration,
                headline="🚗 HIGH CONGESTION: EXTENDING GREEN PHASE",
                action="EXTEND GREEN SIGNAL",
                recommendation=f"Extend green signal by +{green_duration}s to clear {visible_vecs} queued vehicles and prevent junction gridlock.",
                reasons=reasons,
                siren_confidence=siren_conf,
            )

        # Priority 4: Normal Traffic Monitoring
        return UnifiedDecision(
            timestamp_sec=timestamp,
            system_status=STATUS_NORMAL,
            signal_phase=PHASE_GREEN if visible_vecs > 0 else PHASE_RED,
            recommended_duration_sec=15,
            headline="✅ NORMAL TRAFFIC FLOW",
            action="ROUTINE CYCLING",
            recommendation="Traffic and foot-traffic within standard thresholds. Maintaining standard adaptive signal cycle.",
            reasons=[
                f"Vehicles in view: {visible_vecs} ({veh_level})",
                f"Pedestrians in view: {headcount} ({density})",
                "No emergency siren or preemption request",
            ],
            siren_confidence=siren_conf,
        )

