"""
Measures how many frames per second the program is ACTUALLY processing.

This is different from the video's own FPS (e.g. a 30 FPS file may only be
processed at 12 FPS once YOLO is running). The value is averaged over the
last few frames so the number on screen does not jump around.
"""

import time
from collections import deque


class FPSCounter:
    def __init__(self, average_over_frames=30):
        self._frame_times = deque(maxlen=average_over_frames)

    def tick(self):
        """Call once per processed frame."""
        self._frame_times.append(time.perf_counter())

    @property
    def fps(self):
        """Average FPS over the stored frames (0.0 until two frames are seen)."""
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0
