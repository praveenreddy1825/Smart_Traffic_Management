"""
Module 3 demo: track vehicles across frames and count each one ONCE.

Examples:
    python scripts/run_tracking.py                                     # source from config
    python scripts/run_tracking.py --source data/videos/16922729_1920_1080_30fps.mp4
    python scripts/run_tracking.py --save                              # also save annotated MP4
    python scripts/run_tracking.py --no-line                           # disable line counting

On screen:  "#12 car 0.87" = vehicle ID 12.  Yellow lines = recent paths.
Press q / Esc (or close the window) to stop.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from app.detection.vehicle_detector import VehicleDetector  # noqa: E402
from app.detection.visualization import (  # noqa: E402
    draw_counting_line, draw_detections, draw_text_panel, draw_trails)
from app.tracking.vehicle_counter import VehicleCounter  # noqa: E402
from app.utils.config import load_config  # noqa: E402
from app.utils.fps_counter import FPSCounter  # noqa: E402
from app.utils.logger import get_logger, setup_logging  # noqa: E402
from app.video.video_input import VideoInput, VideoSourceError  # noqa: E402

QUIT_KEYS = (ord("q"), 27)


def format_counts(counts, class_names):
    return "  ".join(f"{name}: {counts.get(name, 0)}" for name in class_names)


def main():
    parser = argparse.ArgumentParser(description="Vehicle tracking + counting (Module 3)")
    parser.add_argument("--source", help="video file or webcam index (default: config)")
    parser.add_argument("--no-display", action="store_true", help="do not open a window")
    parser.add_argument("--max-frames", type=int, help="stop after N frames")
    parser.add_argument("--save", action="store_true", help="save annotated video")
    parser.add_argument("--no-line", action="store_true", help="disable line-crossing counting")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)
    logger = get_logger("run_tracking")
    if args.no_line:
        config["tracking"]["counting_line"] = None

    detector = VehicleDetector.from_config(config)
    video = VideoInput.from_config(config, source=args.source)
    try:
        video.open()
    except VideoSourceError as error:
        logger.error("%s", error)
        return 1
    counter = VehicleCounter.from_config(config, video.output_size)
    class_names = detector.class_names
    window_name = config["video"]["window_name"]

    fps_counter = FPSCounter()
    writer = None
    frames_processed = 0
    detections_summed_over_frames = 0  # what naive per-frame counting would report
    start_time = time.perf_counter()
    result = counter.result()

    try:
        for frame in video.frames():
            detections = detector.track(frame.image)
            result = counter.update(detections, frame.index)
            fps_counter.tick()
            frames_processed += 1
            detections_summed_over_frames += len(detections)

            image = frame.image
            draw_counting_line(image, counter.counting_line)
            draw_trails(image, counter.tracks.values())
            draw_detections(image, detections, label_prefix=lambda d: f"#{d.track_id}")
            panel = [
                f"FPS: {fps_counter.fps:5.1f}   Frame: {frame.index}",
                f"UNIQUE vehicles: {result.total_unique}",
                "  " + format_counts(result.unique_by_class, class_names),
                f"Visible now: {result.visible_now}",
            ]
            if counter.counting_line:
                panel.append(f"Crossed line: {result.line_crossings} "
                             f"{result.line_crossings_by_direction or ''}")
            draw_text_panel(image, panel)

            if args.save:
                if writer is None:
                    folder = Path(config["paths"]["test_data_dir"]) / "output"
                    folder.mkdir(parents=True, exist_ok=True)
                    save_path = folder / "tracking_output.mp4"
                    height, width = image.shape[:2]
                    writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             video.fps, (width, height))
                    logger.info("Saving annotated video to %s", save_path)
                writer.write(image)

            if not args.no_display:
                cv2.imshow(window_name, image)
                key = cv2.waitKey(1) & 0xFF
                if key in QUIT_KEYS or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Stopped by user")
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

    elapsed = time.perf_counter() - start_time
    print("\n=== Tracking & counting summary (measured) ===")
    print(f"Frames processed            : {frames_processed}  "
          f"({frames_processed / max(video.fps, 1):.1f} s of video)")
    print(f"Overall FPS                 : {frames_processed / elapsed:.1f}")
    print(f"UNIQUE vehicles counted     : {result.total_unique}")
    for name in class_names:
        print(f"  {name:<12}: {result.unique_by_class.get(name, 0)}")
    if counter.counting_line:
        print(f"Vehicles that crossed line  : {result.line_crossings} "
              f"{result.line_crossings_by_direction}")
    print(f"(Naive per-frame sum would be {detections_summed_over_frames} - "
          f"that is the double counting tracking avoids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
