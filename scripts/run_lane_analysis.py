"""
Module 5 demo: per-lane vehicle counts, density and LOW / MEDIUM / HIGH level.

Examples:
    python scripts/run_lane_analysis.py
    python scripts/run_lane_analysis.py --source data/videos/16922729_1920_1080_30fps.mp4
    python scripts/run_lane_analysis.py --no-display --no-realtime     # fast calibration run

Lanes come from config.yaml -> lanes -> profiles (keyed by video file name).
Draw new ones with scripts/define_lanes.py.
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
    draw_detections, draw_lanes, draw_level_badge, draw_text_panel)
from app.lane.lane_analyzer import LaneAnalyzer  # noqa: E402
from app.tracking.vehicle_counter import VehicleCounter  # noqa: E402
from app.traffic.traffic_analyzer import TrafficAnalyzer  # noqa: E402
from app.utils.config import load_config  # noqa: E402
from app.utils.fps_counter import FPSCounter  # noqa: E402
from app.utils.logger import get_logger, setup_logging  # noqa: E402
from app.video.video_input import VideoInput, VideoSourceError  # noqa: E402

QUIT_KEYS = (ord("q"), 27)


def main():
    parser = argparse.ArgumentParser(description="Lane / region analysis (Module 5)")
    parser.add_argument("--source", help="video file or webcam index (default: config)")
    parser.add_argument("--no-display", action="store_true", help="do not open a window")
    parser.add_argument("--no-realtime", action="store_true", help="process as fast as possible")
    parser.add_argument("--max-frames", type=int, help="stop after N frames")
    parser.add_argument("--save", action="store_true", help="save annotated video")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)
    logger = get_logger("run_lane_analysis")
    # The counting line is not needed here; lanes show the same information more clearly
    config["tracking"]["counting_line"] = None
    source = args.source if args.source is not None else config["video"]["source"]

    detector = VehicleDetector.from_config(config)
    video = VideoInput.from_config(config, source=source)
    if args.no_realtime:
        video.realtime_playback = False
    try:
        video.open()
    except VideoSourceError as error:
        logger.error("%s", error)
        return 1
    lane_analyzer = LaneAnalyzer.from_config(config, source, video.output_size)
    if lane_analyzer is None:
        video.release()
        print("Define lanes first:  python scripts/define_lanes.py --source", source)
        return 1
    counter = VehicleCounter.from_config(config, video.output_size)
    traffic_analyzer = TrafficAnalyzer.from_config(config)
    window_name = config["video"]["window_name"]

    fps_counter = FPSCounter()
    writer = None
    lane_densities = {name: [] for name in lane_analyzer.lane_names}
    lane_level_frames = {name: Counter() for name in lane_analyzer.lane_names}
    lane_reports, traffic_report = [], None
    frames_processed = 0
    start_time = time.perf_counter()

    try:
        for frame in video.frames():
            detections = detector.track(frame.image)
            counting = counter.update(detections, frame.index)
            traffic_report = traffic_analyzer.update(counting, frame.timestamp_sec)
            lane_reports = lane_analyzer.update(detections, frame.timestamp_sec)
            fps_counter.tick()
            frames_processed += 1
            for lane in lane_reports:
                lane_densities[lane.name].append(lane.density)
                lane_level_frames[lane.name][lane.level] += 1

            image = frame.image
            draw_lanes(image, lane_analyzer.polygons, lane_reports)
            draw_detections(image, detections, label_prefix=lambda d: f"#{d.track_id}")
            draw_text_panel(image, [f"FPS: {fps_counter.fps:5.1f}   t = {frame.timestamp_sec:5.1f} s",
                                    f"Unique vehicles: {traffic_report.total_unique}"]
                            + [f"{lane.name}: {lane.visible_now} now, density {lane.density:.1f}"
                               for lane in lane_reports])
            draw_level_badge(image, traffic_report.level)

            if args.save:
                if writer is None:
                    folder = Path(config["paths"]["test_data_dir"]) / "output"
                    folder.mkdir(parents=True, exist_ok=True)
                    height, width = image.shape[:2]
                    writer = cv2.VideoWriter(str(folder / "lane_output.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), video.fps,
                                             (width, height))
                writer.write(image)

            if not args.no_display:
                cv2.imshow(window_name, image)
                key = cv2.waitKey(1) & 0xFF
                if key in QUIT_KEYS or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.max_frames and frames_processed >= args.max_frames:
                break
    except KeyboardInterrupt:
        logger.info("Interrupted with Ctrl+C")
    finally:
        video.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if not lane_reports:
        print("No frames processed.")
        return 1

    elapsed = time.perf_counter() - start_time
    print("\n=== Lane report (end of clip) ===")
    for lane in lane_reports:
        print(f"{lane.text():<36} (density {lane.density:.1f}, "
              f"{lane.unique_vehicles} different vehicles passed)")
    print(f"Whole road: {traffic_report.level} (density {traffic_report.density:.1f})")

    print("\n=== Per-lane calibration data (measured) ===")
    print(f"Frames: {frames_processed}  |  processing FPS: {frames_processed / elapsed:.1f}  |  "
          f"lane thresholds: medium = {lane_analyzer.medium_threshold}, "
          f"high = {lane_analyzer.high_threshold}")
    for name in lane_analyzer.lane_names:
        p10, p50, p90 = np.percentile(lane_densities[name], [10, 50, 90])
        shares = ", ".join(f"{level} {100 * lane_level_frames[name][level] / frames_processed:.0f}%"
                           for level in ("LOW", "MEDIUM", "HIGH"))
        print(f"{name}: density p10/p50/p90/max = {p10:.1f} / {p50:.1f} / {p90:.1f} / "
              f"{max(lane_densities[name]):.1f}  |  {shares}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
