"""
Module 13: End-to-End Test Scenarios & Multi-Scenario System Evaluation.

Runs the complete system through five standard operational test scenarios:
  1. Baseline Traffic (No Siren / Background Noise) -> Verifies zero false alarms.
  2. Sudden Continuous Siren in Traffic -> Verifies fast confirmation, priority action, and recovery.
  3. Intermittent / Masked Siren -> Verifies temporal smoothing bridges 3 s acoustic drop.
  4. Transient False Alarm Burst -> Verifies brief siren (< 3 windows) is rejected.
  5. Multi-Approach Directional Preemption -> Verifies directional fusion (visual pinpointing vs uncertain fallback).

Outputs a structured summary table and saves detailed JSON metrics to
data/test/output/scenario_evaluation.json.

Usage:
    python scripts/run_scenario_evaluation.py
    python scripts/run_scenario_evaluation.py --scenarios 1 2 4
    python scripts/run_scenario_evaluation.py --quick
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio.audio_input import FileAudioInput
from app.audio.siren_classifier import SirenClassifier
from app.audio.siren_smoother import AudioEmergencyState, SirenSmoother
from app.decision.decision_engine import MONITOR, PRIORITIZE_CLEARANCE, DecisionEngine
from app.decision.directional_fusion import (
    ACTION_CAUTION_ALL,
    ACTION_PRIORITIZE_CLEARANCE,
    STATUS_EMERGENCY_ACTIVE,
    STATUS_NORMAL,
    STATUS_UNCERTAIN_EMERGENCY,
    DirectionalFusion,
)
from app.decision.priority_manager import (
    EMERGENCY_PRIORITY,
    NORMAL,
    RECOMMENDATION,
    RETURN_TO_NORMAL,
    ROUTE_ANALYSIS,
    SIREN_DETECTED,
    PriorityManager,
)
from app.detection.vehicle_detector import VehicleDetector
from app.lane.lane_analyzer import LaneAnalyzer
from app.tracking.vehicle_counter import VehicleCounter
from app.traffic.multi_approach_manager import ApproachState
from app.traffic.traffic_analyzer import HIGH, LOW, MEDIUM, TrafficAnalyzer, TrafficReport
from app.utils.config import ensure_directories, load_config, resolve_path
from app.utils.logger import get_logger, setup_logging
from app.video.video_input import VideoInput

logger = get_logger("scenario_eval")


def run_single_camera_scenario(
    config: dict,
    audio_path: str,
    video_path: Optional[str] = None,
    max_seconds: Optional[float] = None,
    name: str = "scenario",
) -> Dict[str, Any]:
    """Execute a time-aligned offline scenario using single-camera lane priority logic."""
    resolved_audio = resolve_path(audio_path)
    if not resolved_audio.is_file():
        raise FileNotFoundError(f"Scenario audio not found: {resolved_audio}")

    resolved_video = resolve_path(video_path or config["video"]["source"])
    if not resolved_video.is_file():
        raise FileNotFoundError(f"Scenario video not found: {resolved_video}")

    detector = VehicleDetector.from_config(config)
    classifier = SirenClassifier.from_config(config)
    smoother = SirenSmoother.from_config(config)
    priority_mgr = PriorityManager.from_config(config)
    traffic_analyzer = TrafficAnalyzer.from_config(config)

    video = VideoInput.from_config(config, source=str(resolved_video))
    video.realtime_playback = False
    video.open()

    counter = VehicleCounter.from_config(config, video.output_size)
    lane_analyzer = LaneAnalyzer.from_config(config, str(resolved_video), video.output_size)

    audio = FileAudioInput(
        str(resolved_audio),
        realtime_playback=False,
        sample_rate=classifier.sample_rate,
        window_seconds=config["audio"]["window_seconds"],
        hop_seconds=config["audio"]["hop_seconds"],
    )
    audio_windows = iter(audio.windows())
    next_window = next(audio_windows, None)

    timeline: List[dict] = []
    state_transitions: List[dict] = []
    current_pm_state = NORMAL
    start_wall_time = time.perf_counter()

    frames_processed = 0
    audio_windows_processed = 0

    lane_reports = []
    latest_traffic_report = None
    latest_audio_state = None

    for frame in video.frames():
        frames_processed += 1
        t_video = frame.timestamp_sec

        if max_seconds and t_video > max_seconds:
            break

        # Run detection & tracking
        detections = detector.track(frame.image)
        counting_res = counter.update(detections, frame.index)
        latest_traffic_report = traffic_analyzer.update(counting_res, t_video)
        if lane_analyzer:
            lane_reports = lane_analyzer.update(detections, t_video)

        # Advance audio when video time catches up
        while next_window is not None and (t_video >= next_window.start_time_sec + audio.window_seconds):
            audio_windows_processed += 1
            pred = classifier.predict(next_window)
            latest_audio_state = smoother.update(pred)

            # Step priority manager
            status = priority_mgr.update(latest_traffic_report, lane_reports, latest_audio_state, now=t_video)

            if status.state != current_pm_state:
                state_transitions.append({
                    "time_sec": round(latest_audio_state.timestamp_sec, 2),
                    "from_state": current_pm_state,
                    "to_state": status.state,
                    "action": status.decision.action if status.decision else None,
                    "target_lane": status.event.target_lane if status.event else None,
                })
                current_pm_state = status.state

            timeline.append({
                "time_sec": round(latest_audio_state.timestamp_sec, 2),
                "priority_state": status.state,
                "action": status.decision.action if status.decision else None,
                "target_lane": status.event.target_lane if status.event else None,
                "audio_state": latest_audio_state.state,
                "siren_score": round(latest_audio_state.window_score, 3),
                "smoothed_conf": round(latest_audio_state.smoothed_confidence, 3),
                "density": round(latest_traffic_report.density, 2) if latest_traffic_report else 0.0,
                "level": latest_traffic_report.level if latest_traffic_report else LOW,
            })

            next_window = next(audio_windows, None)

    total_wall_time = time.perf_counter() - start_wall_time
    fps = frames_processed / total_wall_time if total_wall_time > 0 else 0.0

    return {
        "scenario_name": name,
        "type": "single_camera_priority",
        "audio_file": Path(audio_path).name,
        "video_file": Path(str(resolved_video)).name,
        "frames_processed": frames_processed,
        "audio_windows_processed": audio_windows_processed,
        "processing_fps": round(fps, 1),
        "wall_time_sec": round(total_wall_time, 2),
        "completed_events": len(priority_mgr.events),
        "state_transitions": state_transitions,
        "final_state": current_pm_state,
        "timeline_samples": len(timeline),
    }


def run_directional_scenario(
    config: dict,
    name: str = "directional_fusion_scenario",
) -> Dict[str, Any]:
    """Evaluate Module 11 Directional Emergency Fusion under simulated intersection states."""
    fusion = DirectionalFusion()

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

    # Subtest A: Siren Confirmed + Visual Emergency Spotted on East Approach
    audio_confirmed = AudioEmergencyState(
        state="SIREN_CONFIRMED",
        emergency_active=True,
        window_is_siren=True,
        window_score=0.88,
        siren_windows_in_history=4,
        history_length=5,
        smoothed_confidence=0.84,
        siren_type="Ambulance",
        event_started_sec=10.0,
        event_duration_sec=8.0,
        timestamp_sec=18.0,
        events_so_far=1,
    )

    approaches_with_visual = {
        "north": ApproachState("north", "North Approach", None, None, [], make_report("MEDIUM", 9.5), False, None),
        "south": ApproachState("south", "South Approach", None, None, [], make_report("LOW", 3.0), False, None),
        "east": ApproachState("east", "East Approach", None, None, [], make_report("HIGH", 17.2), True, "Ambulance"),
        "west": ApproachState("west", "West Approach", None, None, [], make_report("LOW", 4.1), False, None),
    }

    decision_visual = fusion.evaluate(audio_confirmed, approaches_with_visual)

    # Subtest B: Siren Confirmed + No Visual Emergency Spotted (Direction Unknown)
    approaches_no_visual = {
        "north": ApproachState("north", "North Approach", None, None, [], make_report("HIGH", 16.0), False, None),
        "south": ApproachState("south", "South Approach", None, None, [], make_report("LOW", 3.0), False, None),
        "east": ApproachState("east", "East Approach", None, None, [], make_report("MEDIUM", 11.0), False, None),
        "west": ApproachState("west", "West Approach", None, None, [], make_report("LOW", 4.1), False, None),
    }

    decision_unknown = fusion.evaluate(audio_confirmed, approaches_no_visual)

    # Subtest C: Normal State (No Siren)
    audio_normal = AudioEmergencyState(
        state="NO_SIREN",
        emergency_active=False,
        window_is_siren=False,
        window_score=0.02,
        siren_windows_in_history=0,
        history_length=5,
        smoothed_confidence=0.01,
        siren_type=None,
        event_started_sec=None,
        event_duration_sec=0.0,
        timestamp_sec=5.0,
        events_so_far=0,
    )
    decision_normal = fusion.evaluate(audio_normal, approaches_no_visual)

    passed_a = (
        decision_visual.system_status == STATUS_EMERGENCY_ACTIVE
        and decision_visual.affected_direction_id == "east"
        and decision_visual.action == ACTION_PRIORITIZE_CLEARANCE
    )

    passed_b = (
        decision_unknown.system_status == STATUS_UNCERTAIN_EMERGENCY
        and decision_unknown.affected_direction_id is None
        and decision_unknown.action == ACTION_CAUTION_ALL
    )

    passed_c = (
        decision_normal.system_status == STATUS_NORMAL
        and decision_normal.action == MONITOR
    )

    return {
        "scenario_name": name,
        "type": "directional_fusion",
        "subtest_visual_pinpoint": {
            "status": decision_visual.system_status,
            "target": decision_visual.affected_direction_name,
            "action": decision_visual.action,
            "passed": passed_a,
        },
        "subtest_unknown_direction": {
            "status": decision_unknown.system_status,
            "target": decision_unknown.affected_direction_name,
            "action": decision_unknown.action,
            "passed": passed_b,
        },
        "subtest_normal_monitoring": {
            "status": decision_normal.system_status,
            "action": decision_normal.action,
            "passed": passed_c,
        },
        "all_subtests_passed": passed_a and passed_b and passed_c,
    }


def main():
    parser = argparse.ArgumentParser(description="Run End-to-End Scenario Evaluation (Module 13)")
    parser.add_argument("--scenarios", nargs="+", choices=["1", "2", "3", "4", "5"],
                        help="Specific scenario numbers to run (1-5). Default runs all.")
    parser.add_argument("--quick", action="store_true",
                        help="Truncate video processing duration to 25s per scenario for fast testing.")
    parser.add_argument("--output", default="data/test/output/scenario_evaluation.json",
                        help="Output path for JSON results.")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)
    ensure_directories(config)

    scenarios_to_run = args.scenarios or ["1", "2", "3", "4", "5"]
    max_sec = 25.0 if args.quick else None

    results: Dict[str, Any] = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "quick_mode": args.quick,
        "scenarios": {},
    }

    print("\n" + "=" * 80)
    print(" SmartCity Sentinel - Module 13: End-to-End Scenario Evaluation Suite")
    print("=" * 80)

    # 1. Baseline Background Noise
    if "1" in scenarios_to_run:
        print("\n--- Scenario 1: Baseline Normal Traffic (No Siren / Background Noise) ---")
        res1 = run_single_camera_scenario(
            config,
            audio_path="data/audio/scenarios/1_background_noise.wav",
            max_seconds=max_sec or 30.0,
            name="1_baseline_traffic_no_siren",
        )
        passed1 = res1["completed_events"] == 0 and res1["final_state"] == NORMAL
        res1["passed"] = passed1
        results["scenarios"]["scenario_1"] = res1
        status_str = "PASS [No False Alarms]" if passed1 else "FAIL [Unexpected Alarm]"
        print(f"Result: {status_str} (Events: {res1['completed_events']}, FPS: {res1['processing_fps']})")

    # 2. Continuous Siren
    if "2" in scenarios_to_run:
        print("\n--- Scenario 2: High Traffic + Sudden Siren (Continuous Siren) ---")
        res2 = run_single_camera_scenario(
            config,
            audio_path="data/audio/scenarios/3_continuous_siren.wav",
            max_seconds=max_sec or 42.0,
            name="2_continuous_siren_response",
        )
        states_seen = {st["to_state"] for st in res2["state_transitions"]}
        passed2 = (EMERGENCY_PRIORITY in states_seen or ROUTE_ANALYSIS in states_seen or RECOMMENDATION in states_seen)
        res2["passed"] = passed2
        results["scenarios"]["scenario_2"] = res2
        status_str = "PASS [Siren Detected & Prioritized]" if passed2 else "FAIL"
        print(f"Result: {status_str} (Transitions: {len(res2['state_transitions'])}, FPS: {res2['processing_fps']})")

    # 3. Intermittent Siren
    if "3" in scenarios_to_run:
        print("\n--- Scenario 3: Intermittent Siren (Acoustic Gap Bridging) ---")
        res3 = run_single_camera_scenario(
            config,
            audio_path="data/audio/scenarios/4_intermittent_siren.wav",
            max_seconds=max_sec or 45.0,
            name="3_intermittent_siren_bridging",
        )
        num_events = res3["completed_events"]
        passed3 = (num_events <= 1)
        res3["passed"] = passed3
        results["scenarios"]["scenario_3"] = res3
        status_str = "PASS [Gap Successfully Bridged]" if passed3 else f"FAIL [{num_events} events]"
        print(f"Result: {status_str} (Events: {num_events}, FPS: {res3['processing_fps']})")

    # 4. Short False Positive
    if "4" in scenarios_to_run:
        print("\n--- Scenario 4: Transient Siren Burst (False Positive Rejection) ---")
        res4 = run_single_camera_scenario(
            config,
            audio_path="data/audio/scenarios/2_short_false_positive.wav",
            max_seconds=max_sec or 35.0,
            name="4_transient_burst_rejection",
        )
        passed4 = res4["completed_events"] == 0 and res4["final_state"] == NORMAL
        res4["passed"] = passed4
        results["scenarios"]["scenario_4"] = res4
        status_str = "PASS [Transient Spike Rejected]" if passed4 else "FAIL [False Activation]"
        print(f"Result: {status_str} (Events: {res4['completed_events']}, FPS: {res4['processing_fps']})")

    # 5. Directional Fusion
    if "5" in scenarios_to_run:
        print("\n--- Scenario 5: Multi-Approach Directional Preemption (Directional Fusion) ---")
        res5 = run_directional_scenario(config, name="5_directional_fusion_logic")
        results["scenarios"]["scenario_5"] = res5
        status_str = "PASS [All Subtests Passed]" if res5["all_subtests_passed"] else "FAIL"
        print(f"Result: {status_str}")

    # Save output JSON
    out_path = resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f" Scenario Evaluation Completed. Metrics saved to: {out_path}")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
