"""Drawing helpers: bounding boxes, labels and small text panels on frames."""

import cv2
import numpy as np

# Fixed BGR colour per class so the same vehicle type always looks the same
CLASS_COLORS = {
    "car": (0, 200, 0),          # green
    "motorcycle": (0, 165, 255), # orange
    "bus": (255, 120, 0),        # blue
    "truck": (180, 0, 180),      # purple
    "person": (0, 220, 220),     # yellow
}
DEFAULT_COLOR = (200, 200, 200)


def draw_detections(image, detections, label_prefix=None):
    """Draw a box and "class confidence" label for each detection (modifies image in place).

    label_prefix: optional function detection -> str placed before the label
                  (used later to show tracking IDs, e.g. "#12 car 0.87").
    """
    for detection in detections:
        color = CLASS_COLORS.get(detection.class_name, DEFAULT_COLOR)
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{detection.class_name} {detection.confidence:.2f}"
        if label_prefix:
            label = f"{label_prefix(detection)} {label}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_top = max(y1 - text_height - baseline - 2, 0)
        cv2.rectangle(image, (x1, label_top),
                      (x1 + text_width + 4, label_top + text_height + baseline + 2), color, -1)
        cv2.putText(image, label, (x1 + 2, label_top + text_height + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def draw_counting_line(image, line, color=(0, 0, 255)):
    """Draw the virtual counting line (line = ((x1, y1), (x2, y2)) in pixels)."""
    if line:
        start, end = (tuple(int(round(v)) for v in point) for point in line)
        cv2.line(image, start, end, color, 2, cv2.LINE_AA)
        cv2.putText(image, "COUNTING LINE", (start[0] + 5, start[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image


def draw_trails(image, tracks, color=(0, 255, 255)):
    """Draw each vehicle's recent path (list of centre points) as a thin line."""
    for track in tracks:
        points = [tuple(int(round(v)) for v in point) for point in track.trail]
        for start, end in zip(points, points[1:]):
            cv2.line(image, start, end, color, 1, cv2.LINE_AA)
    return image


LEVEL_COLORS = {"LOW": (0, 170, 0), "MEDIUM": (0, 140, 255), "HIGH": (0, 0, 220)}


def draw_level_badge(image, level, label="TRAFFIC"):
    """Coloured badge in the top-right corner, e.g. 'TRAFFIC: HIGH' in red."""
    text = f"{label}: {level}"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    right = image.shape[1] - 10
    left = right - text_width - 16
    cv2.rectangle(image, (left, 10), (right, 20 + text_height + baseline),
                  LEVEL_COLORS.get(level, DEFAULT_COLOR), -1)
    cv2.putText(image, text, (left + 8, 15 + text_height), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return image


LANE_OUTLINE_COLOR = (255, 255, 255)


def draw_lanes(image, polygons, lane_reports=None, fill_alpha=0.25):
    """Draw lane polygons. If lane_reports are given, fill each lane with its level colour
    and label it "Lane 1: 5 | MEDIUM"; otherwise just outline and name the lanes.

    polygons: {lane_name: Nx2 array of pixel points}
    """
    reports = {report.name: report for report in (lane_reports or [])}
    overlay = image.copy()
    for name, polygon in polygons.items():
        points = np.round(polygon).astype(np.int32)
        if name in reports:
            cv2.fillPoly(overlay, [points], LEVEL_COLORS.get(reports[name].level, DEFAULT_COLOR))
    if reports:
        cv2.addWeighted(overlay, fill_alpha, image, 1 - fill_alpha, 0, dst=image)

    for name, polygon in polygons.items():
        points = np.round(polygon).astype(np.int32)
        cv2.polylines(image, [points], True, LANE_OUTLINE_COLOR, 2, cv2.LINE_AA)
        label = name
        if name in reports:
            label = f"{name}: {reports[name].visible_now} | {reports[name].level}"
        # Label near the bottom-middle of the lane, where vehicles are largest
        bottom_y = points[:, 1].max() - 12
        center_x = int(points[:, 0].mean())
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        origin = (max(center_x - text_width // 2, 2), int(bottom_y))
        cv2.rectangle(image, (origin[0] - 4, origin[1] - text_height - 6),
                      (origin[0] + text_width + 4, origin[1] + 6), (0, 0, 0), -1)
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return image


def draw_text_panel(image, lines, origin=(10, 25), line_height=24):
    """Write several lines of white text on a semi-transparent dark box (top-left panel).

    The font shrinks automatically if the longest line would not fit the frame
    width (e.g. narrow portrait videos).
    """
    if not lines:
        return image
    x, y = origin
    font, font_scale = cv2.FONT_HERSHEY_SIMPLEX, 0.6
    available_width = image.shape[1] - x - 10
    widest = max(cv2.getTextSize(text, font, font_scale, 1)[0][0] for text in lines)
    if widest > available_width:
        scale_factor = available_width / widest
        font_scale *= scale_factor
        line_height = max(int(line_height * scale_factor), 12)
        widest = available_width

    # Dark box behind the text, blended 60% so the video stays visible
    box_top = max(y - line_height + 4, 0)
    box_bottom = min(y + line_height * (len(lines) - 1) + 8, image.shape[0])
    box_right = min(x + widest + 10, image.shape[1])
    region = image[box_top:box_bottom, max(x - 6, 0):box_right]
    region[:] = (region * 0.4).astype(region.dtype)

    for line_number, text in enumerate(lines):
        position = (x, y + line_height * line_number)
        cv2.putText(image, text, position, font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return image
