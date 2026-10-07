"""
Helper for Module 5: draw lane regions on a video frame, or preview the configured ones.

Draw new lanes (interactive window):
    python scripts/define_lanes.py --source data/videos/13997539_1080_1920_30fps.mp4

    Left click   add a corner point to the current lane
    n / Enter    finish the current lane and start the next one
    u            undo the last point
    s            print the lanes as YAML (paste into config.yaml -> lanes -> profiles)
    q / Esc      quit

Preview the lanes already in config.yaml (saves an image, no window needed):
    python scripts/define_lanes.py --source <clip> --preview
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.detection.visualization import draw_lanes, draw_text_panel  # noqa: E402
from app.lane.lane_analyzer import find_lane_profile, polygon_to_pixels  # noqa: E402
from app.utils.config import load_config  # noqa: E402
from app.utils.logger import setup_logging  # noqa: E402
from app.video.video_input import VideoInput  # noqa: E402


def grab_frame(config, source, frame_number):
    video = VideoInput.from_config(config, source=source)
    video.realtime_playback = False
    frame = None
    for frame in video.frames():
        if frame.index >= frame_number:
            break
    video.release()
    if frame is None:
        raise SystemExit(f"Could not read a frame from {source}")
    return frame.image


def lanes_to_yaml(profile_key, lanes):
    lines = [f'    "{profile_key}":']
    for lane in lanes:
        points = ", ".join(f"[{x:.2f}, {y:.2f}]" for x, y in lane["polygon"])
        lines.append(f'      - name: "{lane["name"]}"')
        lines.append(f"        polygon: [{points}]")
    return "\n".join(lines)


def preview(config, source, image):
    profile_key, lanes = find_lane_profile(config["lanes"], source)
    if not lanes:
        print(f"No lane profile for '{profile_key}' in config.yaml")
        return 1
    height, width = image.shape[:2]
    polygons = {lane["name"]: polygon_to_pixels(lane["polygon"], (width, height)) for lane in lanes}
    draw_lanes(image, polygons)
    folder = Path(config["paths"]["test_data_dir"]) / "output"
    folder.mkdir(parents=True, exist_ok=True)
    save_path = folder / f"lanes_preview_{Path(str(source)).stem}.jpg"
    cv2.imwrite(str(save_path), image)
    print(f"Lane preview saved to {save_path}")
    return 0


def interactive(source, image):
    height, width = image.shape[:2]
    profile_key = "webcam" if str(source).isdigit() else Path(str(source)).name
    finished_lanes = []      # [{"name", "polygon" (fractions)}]
    current_points = []      # pixel points of the lane being drawn

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))

    window = "Define lanes - click corners, n = next lane, s = print YAML, q = quit"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = image.copy()
        polygons = {lane["name"]: polygon_to_pixels(lane["polygon"], (width, height))
                    for lane in finished_lanes}
        draw_lanes(canvas, polygons)
        if current_points:
            points = np.array(current_points, dtype=np.int32)
            cv2.polylines(canvas, [points], False, (0, 255, 255), 2)
            for point in current_points:
                cv2.circle(canvas, point, 4, (0, 255, 255), -1)
        draw_text_panel(canvas, [f"Drawing Lane {len(finished_lanes) + 1}: "
                                 f"{len(current_points)} points",
                                 "n=finish lane  u=undo  s=print YAML  q=quit"])
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("n"), 13):
            if len(current_points) >= 3:
                finished_lanes.append({
                    "name": f"Lane {len(finished_lanes) + 1}",
                    "polygon": [(x / width, y / height) for x, y in current_points],
                })
                current_points.clear()
            else:
                print("A lane needs at least 3 points")
        elif key == ord("u") and current_points:
            current_points.pop()
        elif key == ord("s"):
            print("\n# Paste under config.yaml -> lanes -> profiles:")
            print(lanes_to_yaml(profile_key, finished_lanes))
        elif key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    if finished_lanes:
        print("\n# Final lanes - paste under config.yaml -> lanes -> profiles:")
        print(lanes_to_yaml(profile_key, finished_lanes))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Define or preview lane regions (Module 5)")
    parser.add_argument("--source", help="video file (default: config video.source)")
    parser.add_argument("--frame", type=int, default=0, help="frame number to draw on")
    parser.add_argument("--preview", action="store_true", help="save an image of configured lanes")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)
    source = args.source if args.source is not None else config["video"]["source"]
    image = grab_frame(config, source, args.frame)
    if args.preview:
        return preview(config, source, image)
    return interactive(source, image)


if __name__ == "__main__":
    sys.exit(main())
