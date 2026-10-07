"""
Module 10 demo: emergency priority state machine on a traffic VIDEO + an AUDIO file.

Offline (as fast as possible), video and audio aligned by time, one line per second,
state transitions highlighted, and the emergency event log at the end.

Examples:
    python scripts/run_priority_logic.py --audio data/audio/scenarios/3_continuous_siren.wav
    python scripts/run_priority_logic.py --video data/videos/16922729_1920_1080_30fps.mp4 \
        --audio data/audio/scenarios/3_continuous_siren.wav --audio-offset 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio.audio_input import FileAudioInput  # noqa: E402
from app.audio.siren_classifier import SirenClassifier  # noqa: E402
from app.audio.siren_smoother import SirenSmoother  # noqa: E402
from app.decision.priority_manager import PriorityManager  # noqa: E402
from app.detection.vehicle_detector import VehicleDetector  # noqa: E402
from app.lane.lane_analyzer import LaneAnalyzer  # noqa: E402
from app.tracking.vehicle_counter import VehicleCounter  # noqa: E402
from app.traffic.traffic_analyzer import TrafficAnalyzer  # noqa: E402
from app.utils.config import load_config  # noqa: E402
from app.utils.logger import setup_logging  # noqa: E402
from app.video.video_input import VideoInput  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Emergency priority logic (Module 10)")
    parser.add_argument("--video", help="traffic video (default: config video.source)")
    parser.add_argument("--audio", required=True, help="audio file to pair with the video")
    parser.add_argument("--audio-offset", type=float, default=0.0,
                        help="skip the first N seconds of the audio")
    parser.add_argument("--emergency-lane", help="override decision.emergency_lane")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)
    config["tracking"]["counting_line"] = None
    if args.emergency_lane:
        config["decision"]["emergency_lane"] = args.emergency_lane
    video_source = args.video or config["video"]["source"]

    detector = VehicleDetector.from_config(config)
    classifier = SirenClassifier.from_config(config)
    smoother = SirenSmoother.from_config(config)
    manager = PriorityManager.from_config(config)
    traffic_analyzer = TrafficAnalyzer.from_config(config)

    video = VideoInput.from_config(config, source=video_source)
    video.realtime_playback = False
    video.open()
    counter = VehicleCounter.from_config(config, video.output_size)
    lane_analyzer = LaneAnalyzer.from_config(config, video_source, video.output_size)
    audio = FileAudioInput(args.audio, realtime_playback=False,
                           sample_rate=classifier.sample_rate,
                           window_seconds=config["audio"]["window_seconds"],
                           hop_seconds=config["audio"]["hop_seconds"])
    audio_windows = (w for w in audio.windows() if w.start_time_sec >= args.audio_offset)
    next_window = next(audio_windows, None)

    print(f"\nVideo: {Path(str(video_source)).name} | Audio: {Path(args.audio).name} "
          f"(offset {args.audio_offset:g} s)")
    print(f"{'t':>5}  {'priority state':<18} {'active':<6} {'lanes (level+density)':<24} "
          f"{'action':<20} {'target':<7} event conf")
    lane_reports, status, seen_transitions = [], None, 0
    for frame in video.frames():
        detections = detector.track(frame.image)
        traffic_report = traffic_analyzer.update(counter.update(detections, frame.index),
                                                 frame.timestamp_sec)
        if lane_analyzer:
            lane_reports = lane_analyzer.update(detections, frame.timestamp_sec)

        while next_window is not None and (frame.timestamp_sec >= next_window.start_time_sec
                                           - args.audio_offset + audio.window_seconds):
            next_window.start_time_sec -= args.audio_offset
            audio_state = smoother.update(classifier.predict(next_window))
            status = manager.update(traffic_report, lane_reports, audio_state,
                                    now=frame.timestamp_sec)
            for when, old, new, reason in status.transitions[seen_transitions:]:
                print(f"       ** {when:>5.1f}s  {old} -> {new}: {reason}")
            seen_transitions = len(status.transitions)
            event = status.event if status.priority_active else None
            lanes_text = " ".join(f"{l.name[-1]}:{l.level[0]}{l.density:.1f}"
                                  for l in lane_reports)
            print(f"{frame.timestamp_sec:>5.1f}  {status.state:<18} "
                  f"{'YES' if status.priority_active else '-':<6} {lanes_text:<24} "
                  f"{status.decision.action:<20} {status.decision.target_lane or '-':<7} "
                  f"{f'{event.mean_confidence:.2f} (peak {event.peak_confidence:.2f})' if event else '-'}")
            next_window = next(audio_windows, None)
        if next_window is None:
            break
    video.release()

    if status:
        print(f"\nLast message: {status.message}")
    print("\n=== Emergency event log ===")
    for event in manager.events:
        end = f"{event.ended_at_sec:.1f}s" if event.ended_at_sec is not None else "still active"
        print(f"Event {event.event_id}: confirmed {event.confirmed_at_sec:.1f}s, ended {end}, "
              f"target {event.target_lane or 'whole road'} ({event.target_rule}), "
              f"confidence mean {event.mean_confidence:.2f} / peak {event.peak_confidence:.2f}, "
              f"type {event.siren_type}, actions {dict(event.actions)}")
        for switch in event.target_switches:
            print(f"   target switch {switch}")
    if not manager.events:
        print("No emergency events.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
