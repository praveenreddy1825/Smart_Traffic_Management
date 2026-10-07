"""
Module 2 demo: run YOLO vehicle detection on an image, a video or the webcam.

Examples:
    python scripts/run_detection.py --source data/videos/traffic.mp4
    python scripts/run_detection.py --source 0                         # webcam
    python scripts/run_detection.py --source photo.jpg                 # single image
    python scripts/run_detection.py --source data/videos/traffic.mp4 --conf 0.5 --save

Images: the annotated result is saved to data/test/output/.
Videos: press q / Esc (or close the window) to stop. --save writes an annotated MP4.

NOTE: the counts shown are vehicles visible IN THE CURRENT FRAME.
Unique vehicle counting (no double counting) is Module 3.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from app.detection.vehicle_detector import VehicleDetector, count_by_class  # noqa: E402
from app.detection.visualization import draw_detections, draw_text_panel  # noqa: E402
from app.utils.config import load_config  # noqa: E402
from app.utils.fps_counter import FPSCounter  # noqa: E402
from app.utils.logger import get_logger, setup_logging  # noqa: E402
from app.video.video_input import VideoInput, VideoSourceError  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
QUIT_KEYS = (ord("q"), 27)


def output_dir(config):
    folder = Path(config["paths"]["test_data_dir"]) / "output"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def run_on_image(detector, image_path, config, show):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return 1

    start = time.perf_counter()
    detections = detector.detect(image)
    inference_ms = (time.perf_counter() - start) * 1000

    print(f"\n{len(detections)} detection(s) in {Path(image_path).name} "
          f"(inference {inference_ms:.1f} ms, includes first-call warm-up):")
    for detection in detections:
        x1, y1, x2, y2 = (int(v) for v in detection.bbox)
        print(f"  {detection.class_name:<10} conf={detection.confidence:.2f}  "
              f"box=({x1},{y1})-({x2},{y2})")
    print(f"Per-class: {count_by_class(detections)}")

    draw_detections(image, detections)
    save_path = output_dir(config) / f"{Path(image_path).stem}_detected.jpg"
    cv2.imwrite(str(save_path), image)
    print(f"Annotated image saved to {save_path}")

    if show:
        cv2.imshow("Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_on_video(detector, config, source, show, max_frames, save):
    logger = get_logger("run_detection")
    video = VideoInput.from_config(config, source=source)
    window_name = config["video"]["window_name"]
    fps_counter = FPSCounter()
    writer = None

    frames_processed = 0
    total_inference_s = 0.0
    total_detections = 0
    confidences = []
    start_time = time.perf_counter()

    try:
        for frame in video.frames():
            inference_start = time.perf_counter()
            detections = detector.detect(frame.image)
            inference_s = time.perf_counter() - inference_start

            fps_counter.tick()
            frames_processed += 1
            if frames_processed > 1:  # skip first frame: it includes model warm-up
                total_inference_s += inference_s
            total_detections += len(detections)
            confidences.extend(d.confidence for d in detections)

            draw_detections(frame.image, detections)
            counts = count_by_class(detections)
            draw_text_panel(frame.image, [
                f"FPS: {fps_counter.fps:5.1f} | inference: {inference_s * 1000:5.1f} ms",
                f"In this frame: {len(detections)} vehicles",
                "  ".join(f"{name}: {counts.get(name, 0)}" for name in detector.class_names),
            ])

            if save:
                if writer is None:
                    height, width = frame.image.shape[:2]
                    save_path = output_dir(config) / "detection_output.mp4"
                    writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             video.fps, (width, height))
                    logger.info("Saving annotated video to %s", save_path)
                writer.write(frame.image)

            if show:
                cv2.imshow(window_name, frame.image)
                key = cv2.waitKey(1) & 0xFF
                if key in QUIT_KEYS or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Stopped by user")
                    break
            if max_frames and frames_processed >= max_frames:
                break
    except VideoSourceError as error:
        logger.error("%s", error)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted with Ctrl+C")
    finally:
        video.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    timed_frames = max(frames_processed - 1, 1)
    print("\n=== Detection summary (measured) ===")
    print(f"Frames processed      : {frames_processed}")
    print(f"Overall FPS           : {frames_processed / elapsed:.1f}  (includes model warm-up)")
    print(f"Avg inference / frame : {total_inference_s / timed_frames * 1000:.1f} ms  (first frame excluded)")
    print(f"Avg detections / frame: {total_detections / max(frames_processed, 1):.1f}")
    if confidences:
        print(f"Mean confidence       : {sum(confidences) / len(confidences):.2f}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Vehicle detection test (Module 2)")
    parser.add_argument("--source", help="image, video file or webcam index (default: config)")
    parser.add_argument("--conf", type=float, help="override confidence threshold")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], help="override device")
    parser.add_argument("--no-display", action="store_true", help="do not open a window")
    parser.add_argument("--max-frames", type=int, help="stop after N frames")
    parser.add_argument("--save", action="store_true", help="save annotated video")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)
    if args.conf is not None:
        config["detection"]["confidence_threshold"] = args.conf
    if args.device:
        config["device"] = args.device

    detector = VehicleDetector.from_config(config)
    source = args.source if args.source is not None else config["video"]["source"]

    if Path(str(source)).suffix.lower() in IMAGE_EXTENSIONS:
        return run_on_image(detector, source, config, show=not args.no_display)
    return run_on_video(detector, config, source, show=not args.no_display,
                        max_frames=args.max_frames, save=args.save)


if __name__ == "__main__":
    sys.exit(main())
