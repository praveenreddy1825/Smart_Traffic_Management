"""
Module 2 - Vehicle Detection with a pretrained YOLO model.

    detector = VehicleDetector.from_config(config)
    detections = detector.detect(frame.image)
    for d in detections:
        d.class_name, d.confidence, d.bbox, d.center

The model is pretrained on the COCO dataset (80 everyday object classes).
We only keep the traffic-related ones (car, motorcycle, bus, truck).
No training is done in this project.
"""

from dataclasses import dataclass
from pathlib import Path

from app.utils.config import resolve_path
from app.utils.device import get_device
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    bbox: tuple            # (x1, y1, x2, y2) pixel corners: top-left and bottom-right
    class_id: int          # COCO class id (e.g. 2 = car)
    class_name: str        # e.g. "car"
    confidence: float      # model confidence, 0.0 - 1.0
    track_id: int | None = None  # same number for the same vehicle across frames (Module 3)

    @property
    def center(self):
        """Centre point of the box - used later to decide which lane a vehicle is in."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class VehicleDetector:
    def __init__(
        self,
        weights_path,
        class_names,
        confidence_threshold=0.35,
        iou_threshold=0.5,
        image_size=640,
        device="auto",
        tracker_config="bytetrack.yaml",
        tracking_confidence=0.1,
    ):
        # Imported here so the rest of the project can be used without loading torch
        from ultralytics import YOLO

        self.weights_path = Path(weights_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = get_device(device)
        self.tracker_config = str(tracker_config)
        self.tracking_confidence = tracking_confidence

        # If the file is missing, Ultralytics downloads the official weights to this path
        logger.info("Loading YOLO model %s on %s", self.weights_path.name, self.device)
        self.model = YOLO(str(self.weights_path))

        # The model knows its own labels: {0: "person", 1: "bicycle", 2: "car", ...}
        name_to_id = {name: class_id for class_id, name in self.model.names.items()}
        unknown = [name for name in class_names if name not in name_to_id]
        if unknown:
            raise ValueError(f"Classes not in the model's labels: {unknown}. "
                             f"Valid examples: car, motorcycle, bus, truck, person")
        self.class_names = list(class_names)
        self.class_ids = [name_to_id[name] for name in class_names]
        logger.info("Detecting classes %s (COCO ids %s), confidence >= %.2f",
                    self.class_names, self.class_ids, self.confidence_threshold)

    @classmethod
    def from_config(cls, config):
        settings = config["detection"]
        class_names = list(settings["vehicle_classes"])
        if settings.get("include_person", False) and "person" not in class_names:
            class_names.append("person")
        tracking = config.get("tracking", {})
        return cls(
            weights_path=Path(config["paths"]["models_dir"]) / settings["model_name"],
            class_names=class_names,
            confidence_threshold=settings.get("confidence_threshold", 0.35),
            iou_threshold=settings.get("iou_threshold", 0.5),
            image_size=settings.get("image_size", 640),
            device=config.get("device", "auto"),
            tracker_config=resolve_path(tracking.get("tracker_config", "config/bytetrack.yaml")),
            tracking_confidence=tracking.get("detection_confidence", 0.1),
        )

    def detect(self, image):
        """Run YOLO on one BGR image and return a list of Detection objects.

        Ultralytics does the standard steps internally:
          resize/pad to image_size -> neural network forward pass ->
          drop boxes below confidence_threshold -> remove duplicate
          overlapping boxes (NMS, iou_threshold) -> keep only our classes.
        """
        result = self.model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )[0]
        return self._to_detections(result)

    def track(self, image):
        """Detect AND track: like detect(), but each Detection also gets a track_id.

        Uses ByteTrack (built into Ultralytics). It remembers vehicles between
        calls, so frames must be passed in order from ONE video. Call
        reset_tracking() before starting a different video.

        Per frame, ByteTrack:
          1. predicts where every known vehicle should be now (Kalman filter),
          2. matches confident new boxes to those predictions by box overlap (IoU),
          3. tries to match remaining weak boxes to still-unmatched tracks,
          4. starts new IDs for unmatched confident boxes; drops tracks lost too long.
        """
        result = self.model.track(
            image,
            persist=True,                     # keep tracker state between frames
            tracker=self.tracker_config,
            conf=self.tracking_confidence,    # low on purpose - see config "tracking"
            iou=self.iou_threshold,
            imgsz=self.image_size,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )[0]
        # Only boxes that belong to a track get an ID; untracked weak boxes are dropped
        return [d for d in self._to_detections(result) if d.track_id is not None]

    def reset_tracking(self):
        """Forget all tracks (IDs restart) - use when switching to another video."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()

    def _to_detections(self, result):
        """Convert an Ultralytics result into our simple Detection objects."""
        track_ids = result.boxes.id
        detections = []
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                class_name=self.model.names[class_id],
                confidence=float(box.conf.item()),
                track_id=int(track_ids[index].item()) if track_ids is not None else None,
            ))
        return detections


def count_by_class(detections):
    """Number of detections per class IN ONE FRAME (not unique vehicles - see Module 3)."""
    counts = {}
    for detection in detections:
        counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
    return counts
