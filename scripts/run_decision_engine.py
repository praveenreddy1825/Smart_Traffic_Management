"""
Module 9 demo: fuse a traffic VIDEO with an AUDIO file and print the decision every second.

Runs offline (as fast as possible) and keeps video and audio aligned by time:
an audio window is used once the video has reached that window's end time.

Examples:
    python scripts/run_decision_engine.py --audio data/audio/scenarios/1_background_noise.wav
    python scripts/run_decision_engine.py --audio data/audio/scenarios/3_continuous_siren.wav
    python scripts/run_decision_engine.py --video data/videos/16922729_1920_1080_30fps.mp4 \
        --audio data/audio/scenarios/3_continuous_siren.wav --audio-offset 8

--audio-offset N skips the first N seconds of the audio (useful for short videos).
The live, real-time combination of camera + microphone is Module 12.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio.audio_input import FileAudioInput  # noqa: E402
from app.audio.siren_classifier import SirenClassifier  # noqa: E402
from app.audio.siren_smoother import SirenSmoother  # noqa: E402
from app.decision.decision_engine import DecisionEngine  # noqa: E402
from app.detection.vehicle_detector import VehicleDetector  # noqa: E402
from app.lane.lane_analyzer import LaneAnalyzer  # noqa: E402
from app.tracking.vehicle_counter import VehicleCounter  # noqa: E402
from app.traffic.traffic_analyzer import TrafficAnalyzer  # noqa: E402
from app.utils.config import load_config  # noqa: E402
from app.utils.logger import setup_logging  # noqa: E402
from app.video.video_input import VideoInput  # noqa: E402


def lane_summary(lane_reports):
    return " ".join(f"{lane.name[-1]}:{lane.level[0]}{lane.density:.1f}" for lane in lane_reports)


def main():
    parser = argparse.ArgumentParser(description="Multimodal decision engine (Module 9)")
    parser.add_argument("--video", help="traffic video (default: config video.source)")
    parser.add_argument("--audio", required=True, help="audio file to pair with the video")
    parser.add_argument("--audio-offset", type=float, default=0.0,
                        help="skip the first N seconds of the audio")
    parser.add_argument("--emergency-lane", help="override decision.emergency_lane")
    parser.add_argument("--verbose", action="store_true", help="print full reasons at changes")
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
    engine = DecisionEngine.from_config(config)
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

    print(f"\nVideo: {Path(str(video_source)).name}  |  Audio: {Path(args.audio).name} "
          f"(offset {args.audio_offset:g} s)  |  emergency_lane: "
          f"{config['decision']['emergency_lane'] or 'most congested'}")
    print(f"{'t':>5}  {'siren':<16} {'conf':>4}  {'traffic':<7} {'lanes':<22} {'mode':<9} "
          f"{'action':<20} target")

    action_seconds = Counter()
    lane_reports, previous_key = [], None
    for frame in video.frames():
        detections = detector.track(frame.image)
        counting = counter.update(detections, frame.index)
        traffic_report = traffic_analyzer.update(counting, frame.timestamp_sec)
        if lane_analyzer:
            lane_reports = lane_analyzer.update(detections, frame.timestamp_sec)

        # Use every audio window whose END time the video has now reached
        while next_window is not None and (frame.timestamp_sec >= next_window.start_time_sec
                                           - args.audio_offset + audio.window_seconds):
            next_window.start_time_sec -= args.audio_offset   # align audio time with video time
            audio_state = smoother.update(classifier.predict(next_window))
            decision = engine.decide(traffic_report, lane_reports, audio_state)
            action_seconds[decision.action] += 1
            print(f"{frame.timestamp_sec:>5.1f}  {audio_state.state:<16} "
                  f"{audio_state.smoothed_confidence:>4.2f}  {traffic_report.level:<7} "
                  f"{lane_summary(lane_reports):<22} {decision.mode:<9} {decision.action:<20} "
                  f"{decision.target_lane or '-'}")
            key = (decision.mode, decision.action, decision.target_lane)
            if key != previous_key and (args.verbose or decision.emergency_active):
                print("       >> " + decision.text().replace("\n", "\n          "))
            previous_key = key
            next_window = next(audio_windows, None)
        if next_window is None:
            break

    video.release()
    print(f"\nSeconds per action: {dict(action_seconds)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
