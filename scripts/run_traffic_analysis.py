"""
Module 4 demo: traffic density and LOW / MEDIUM / HIGH level, plus threshold calibration data.

Examples:
    python scripts/run_traffic_analysis.py
    python scripts/run_traffic_analysis.py --source data/videos/16922729_1920_1080_30fps.mp4
    python scripts/run_traffic_analysis.py --source <clip> --no-display --no-realtime   # fast calibration run

At the end it prints the brief-style report AND the measured density
distribution (percentiles). Use those percentiles to calibrate
traffic.level_thresholds in config.yaml for a camera:
put "medium" above typical free-flowing density and "high" at the density
seen when the road is clearly congested.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.detection.vehicle_detector import VehicleDetector  # noqa: E402
from app.detection.visualization import (  # noqa: E402
    draw_counting_line, draw_detections, draw_level_badge, draw_text_panel)
from app.tracking.vehicle_counter import VehicleCounter  # noqa: E402
from app.traffic.traffic_analyzer import TrafficAnalyzer  # noqa: E402
from app.utils.config import load_config  # noqa: E402
from app.utils.fps_counter import FPSCounter  # noqa: E402
from app.utils.logger import get_logger, setup_logging  # noqa: E402
from app.video.video_input import VideoInput, VideoSourceError  # noqa: E402

QUIT_KEYS = (ord("q"), 27)


def main():
    parser = argparse.ArgumentParser(description="Traffic analysis (Module 4)")
    parser.add_argument("--source", help="video file or webcam index (default: config)")
    parser.add_argument("--no-display", action="store_true", help="do not open a window")
    parser.add_argument("--no-realtime", action="store_true", help="process as fast as possible")
    parser.add_argument("--max-frames", type=int, help="stop after N frames")
    parser.add_argument("--save", action="store_true", help="save annotated video")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)
    logger = get_logger("run_traffic_analysis")

    detector = VehicleDetector.from_config(config)
    video = VideoInput.from_config(config, source=args.source)
    if args.no_realtime:
        video.realtime_playback = False
    try:
        video.open()
    except VideoSourceError as error:
        logger.error("%s", error)
        return 1
    counter = VehicleCounter.from_config(config, video.output_size)
    analyzer = TrafficAnalyzer.from_config(config)
    window_name = config["video"]["window_name"]

    fps_counter = FPSCounter()
    writer = None
    densities = []
    level_frames = Counter()
    report = None
    start_time = time.perf_counter()

    try:
        for frame in video.frames():
            detections = detector.track(frame.image)
            counting = counter.update(detections, frame.index)
            report = analyzer.update(counting, frame.timestamp_sec)
            fps_counter.tick()
            densities.append(report.density)
            level_frames[report.level] += 1

            image = frame.image
            draw_counting_line(image, counter.counting_line)
            draw_detections(image, detections, label_prefix=lambda d: f"#{d.track_id}")
            draw_text_panel(image, [
                f"FPS: {fps_counter.fps:5.1f}   t = {frame.timestamp_sec:5.1f} s",
                f"Unique vehicles: {report.total_unique}",
                f"In view: {report.visible_now}   load: {report.load_now:.1f}",
                f"Density ({analyzer.density_window_seconds:.0f}s avg): {report.density:.1f}",
                f"Flow: {report.flow_per_minute:.0f} veh/min",
            ], origin=(10, 25))
            draw_level_badge(image, report.level)

            if args.save:
                if writer is None:
                    folder = Path(config["paths"]["test_data_dir"]) / "output"
                    folder.mkdir(parents=True, exist_ok=True)
                    height, width = image.shape[:2]
                    writer = cv2.VideoWriter(str(folder / "traffic_output.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), video.fps,
                                             (width, height))
                writer.write(image)

            if not args.no_display:
                cv2.imshow(window_name, image)
                key = cv2.waitKey(1) & 0xFF
                if key in QUIT_KEYS or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.max_frames and len(densities) >= args.max_frames:
                break
    except KeyboardInterrupt:
        logger.info("Interrupted with Ctrl+C")
    finally:
        video.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if report is None:
        print("No frames processed.")
        return 1

    elapsed = time.perf_counter() - start_time
    total_frames = len(densities)
    print("\n=== Traffic report (end of clip) ===")
    print(report.summary_text(detector.class_names))

    print("\n=== Calibration data (measured on this clip) ===")
    print(f"Frames: {total_frames}  |  processing FPS: {total_frames / elapsed:.1f}")
    percentiles = np.percentile(densities, [10, 25, 50, 75, 90])
    print("Density percentiles  p10 / p25 / p50 / p75 / p90 / max : "
          + " / ".join(f"{value:.1f}" for value in percentiles) + f" / {max(densities):.1f}")
    print(f"Current thresholds: medium = {analyzer.medium_threshold}, "
          f"high = {analyzer.high_threshold}")
    print("Share of time per level: " + ", ".join(
        f"{level} {100 * level_frames[level] / total_frames:.0f}%"
        for level in ("LOW", "MEDIUM", "HIGH")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
