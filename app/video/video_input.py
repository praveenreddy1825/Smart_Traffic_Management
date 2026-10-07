"""
Module 1 - Video Input.

One class, VideoInput, gives the rest of the system frames from EITHER a
recorded video file OR a webcam, in exactly the same format:

    with VideoInput.from_config(config) as video:
        for frame in video.frames():
            frame.image          # the picture (NumPy array, BGR colour order)
            frame.index          # frame number: 0, 1, 2, ...
            frame.timestamp_sec  # time position in seconds

Later modules (YOLO detection etc.) only ever see VideoFrame objects, so they
do not care where the video comes from.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.utils.config import resolve_path
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VideoSourceError(Exception):
    """Raised when a video file or webcam cannot be opened."""


@dataclass
class VideoFrame:
    image: np.ndarray      # frame pixels, shape (height, width, 3), BGR
    index: int             # frame number since start (keeps counting if the video loops)
    timestamp_sec: float   # video time for files, wall-clock time since start for webcams


def parse_source(source):
    """Convert a config value into a webcam index (int) or a file Path.

    0, "0", "1" ... -> webcam index
    anything else   -> file path (relative paths start at the project root)
    """
    if isinstance(source, int):
        return source
    source_text = str(source).strip()
    if source_text.isdigit():
        return int(source_text)
    return resolve_path(source_text)


class VideoInput:
    def __init__(
        self,
        source,
        max_frame_size=None,
        realtime_playback=True,
        loop_video=False,
        webcam_fallback_fps=30,
        max_consecutive_read_failures=30,
    ):
        self.source = parse_source(source)
        self.is_webcam = isinstance(self.source, int)
        self.max_frame_size = max_frame_size
        self.realtime_playback = realtime_playback
        self.loop_video = loop_video
        self.webcam_fallback_fps = webcam_fallback_fps
        self.max_consecutive_read_failures = max_consecutive_read_failures

        # Filled in by open()
        self.fps = 0.0                 # frames per second of the SOURCE
        self.frame_count = None        # total frames (video files only)
        self.original_size = None      # (width, height) from the source
        self.output_size = None        # (width, height) after resizing

        self._capture = None
        self._next_frame_index = 0
        self._frames_read_this_pass = 0   # used to detect an unreadable/empty file when looping
        self._start_time = 0.0            # set to the real start time in open()

    @classmethod
    def from_config(cls, config, source=None):
        """Create a VideoInput from the 'video' section of the config.

        `source` overrides the configured source (e.g. from a command-line flag).
        """
        video_settings = config["video"]
        return cls(
            source=video_settings["source"] if source is None else source,
            max_frame_size=video_settings.get("max_frame_size"),
            realtime_playback=video_settings.get("realtime_playback", True),
            loop_video=video_settings.get("loop_video", False),
            webcam_fallback_fps=video_settings.get("webcam_fallback_fps", 30),
            max_consecutive_read_failures=video_settings.get("max_consecutive_read_failures", 30),
        )

    # ------------------------------------------------------------------ open / close

    def open(self):
        """Open the file/webcam and read its properties. Raises VideoSourceError on failure."""
        if isinstance(self.source, int):  # webcam
            # DirectShow opens webcams much faster than the default backend on Windows
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            self._capture = cv2.VideoCapture(self.source, backend)
            description = f"webcam {self.source}"
        else:
            if not Path(self.source).is_file():
                raise VideoSourceError(f"Video file not found: {self.source}")
            self._capture = cv2.VideoCapture(str(self.source))
            description = f"file {Path(self.source).name}"

        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise VideoSourceError(f"Could not open {description}")

        reported_fps = self._capture.get(cv2.CAP_PROP_FPS)
        # Webcams often report 0 (or nonsense); fall back to the configured value
        if reported_fps is None or reported_fps <= 0 or reported_fps > 240:
            reported_fps = self.webcam_fallback_fps
            logger.warning("Source did not report a valid FPS - using %s", reported_fps)
        self.fps = float(reported_fps)

        if not self.is_webcam:
            self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_size = (width, height)
        self.output_size = self._resized_size(width, height)

        self._next_frame_index = 0
        self._frames_read_this_pass = 0
        self._start_time = time.perf_counter()

        logger.info(
            "Opened %s | %dx%d -> %dx%d | %.1f FPS | frames: %s",
            description, width, height, *self.output_size, self.fps,
            self.frame_count if self.frame_count else "live",
        )
        return self

    def release(self):
        """Free the file/camera. Safe to call more than once."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Video source released after %d frames", self._next_frame_index)

    @property
    def is_opened(self):
        return self._capture is not None and self._capture.isOpened()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc_info):
        self.release()

    # ------------------------------------------------------------------ reading

    def read(self):
        """Return the next VideoFrame, or None when the stream has ended."""
        capture = self._capture
        if capture is None or not capture.isOpened():
            raise VideoSourceError("Video source is not open - call open() first")

        consecutive_failures = 0
        while True:
            success, image = capture.read()
            if success:
                break

            if not self.is_webcam:
                # End of file: restart if looping (and the file actually had frames)
                if self.loop_video and self._frames_read_this_pass > 0:
                    logger.info("End of video reached - looping")
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._frames_read_this_pass = 0
                    continue
                return None

            # Webcam: a dropped frame is normal; only give up after many in a row
            consecutive_failures += 1
            if consecutive_failures >= self.max_consecutive_read_failures:
                logger.error("Webcam returned no frames %d times in a row - stopping",
                             consecutive_failures)
                return None
            time.sleep(0.01)

        frame_index = self._next_frame_index
        self._next_frame_index += 1
        self._frames_read_this_pass += 1

        if self.is_webcam:
            timestamp_sec = time.perf_counter() - self._start_time
        else:
            timestamp_sec = frame_index / self.fps
            if self.realtime_playback:
                self._wait_until(timestamp_sec)

        return VideoFrame(image=self._resize(image), index=frame_index, timestamp_sec=timestamp_sec)

    def frames(self):
        """Generator over all frames; opens the source if needed and always releases it."""
        if not self.is_opened:
            self.open()
        try:
            while True:
                frame = self.read()
                if frame is None:
                    break
                yield frame
        finally:
            self.release()

    # ------------------------------------------------------------------ helpers

    def _wait_until(self, timestamp_sec):
        """Sleep so a recorded video is not delivered faster than its real speed.

        If processing is SLOWER than real time we never sleep - the video simply
        plays slower (frames are not skipped).
        """
        delay = (self._start_time + timestamp_sec) - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    def _resized_size(self, width, height):
        """Shrink so the LONGER side is at most max_frame_size, keeping the aspect ratio.

        Works the same for landscape (1920x1080 -> 960x540) and portrait
        (1080x1920 -> 540x960) videos. Small frames are never enlarged:
        that adds no detail but slows detection down.
        """
        longer_side = max(width, height)
        if not self.max_frame_size or longer_side == 0 or longer_side <= self.max_frame_size:
            return (width, height)
        scale = self.max_frame_size / longer_side
        return (round(width * scale), round(height * scale))

    def _resize(self, image):
        height, width = image.shape[:2]
        new_width, new_height = self._resized_size(width, height)
        if (new_width, new_height) == (width, height):
            return image
        # INTER_AREA gives the best quality when shrinking an image
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
