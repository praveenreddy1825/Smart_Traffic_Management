"""
Module 1 demo: read a traffic video (or the webcam) and show it in a window.

Examples:
    python scripts/run_video_input.py                                  # source from config
    python scripts/run_video_input.py --source data/videos/traffic.mp4
    python scripts/run_video_input.py --source 0                       # webcam
    python scripts/run_video_input.py --source 0 --no-display --max-frames 100

Press  q  or  Esc  (or close the window) to stop.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from app.utils.config import load_config  # noqa: E402
from app.utils.fps_counter import FPSCounter  # noqa: E402
from app.utils.logger import get_logger, setup_logging  # noqa: E402
from app.video.video_input import VideoInput, VideoSourceError  # noqa: E402

QUIT_KEYS = (ord("q"), 27)  # q or Esc


def draw_overlay(image, frame, processing_fps, source_fps):
    lines = [
        f"Frame: {frame.index}",
        f"Time: {frame.timestamp_sec:6.2f} s",
        f"Processing FPS: {processing_fps:5.1f}  (source {source_fps:.1f})",
    ]
    for line_number, text in enumerate(lines):
        position = (10, 25 + 25 * line_number)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="Video input test (Module 1)")
    parser.add_argument("--source", help="video file path or webcam index (overrides config)")
    parser.add_argument("--no-display", action="store_true", help="do not open a window")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    parser.add_argument("--no-realtime", action="store_true",
                        help="process a file as fast as possible instead of at real speed")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)
    logger = get_logger("run_video_input")

    video = VideoInput.from_config(config, source=args.source)
    if args.no_realtime:
        video.realtime_playback = False
    window_name = config["video"]["window_name"]

    fps_counter = FPSCounter()
    frames_shown = 0
    start_time = time.perf_counter()

    try:
        for frame in video.frames():
            fps_counter.tick()
            frames_shown += 1

            if not args.no_display:
                draw_overlay(frame.image, frame, fps_counter.fps, video.fps)
                cv2.imshow(window_name, frame.image)
                key = cv2.waitKey(1) & 0xFF
                window_closed = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
                if key in QUIT_KEYS or window_closed:
                    logger.info("Stopped by user")
                    break

            if args.max_frames and frames_shown >= args.max_frames:
                logger.info("Reached --max-frames=%d", args.max_frames)
                break
    except VideoSourceError as error:
        logger.error("%s", error)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted with Ctrl+C")
    finally:
        video.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    average_fps = frames_shown / elapsed if elapsed > 0 else 0.0
    print(f"\nFrames read: {frames_shown} | Elapsed: {elapsed:.2f} s | "
          f"Average processing FPS: {average_fps:.1f} | Output size: {video.output_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
