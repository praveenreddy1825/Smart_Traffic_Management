"""
Module: ScenarioStream - Dual Video & Audio Demuxer for Scenario-Driven Traffic AI.

Simultaneously streams:
  1. High-resolution video frames (OpenCV VideoCapture)
  2. Synchronized 16kHz mono audio windows (extracted via PyAV from MP4 or companion WAV)
     for direct inference by the AST Siren Classifier.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple
import time

import av
import cv2
import numpy as np
import soundfile as sf

from app.audio.audio_input import AudioWindow
from app.utils.config import resolve_path
from app.utils.logger import get_logger
from app.video.video_input import VideoFrame, VideoSourceError

logger = get_logger(__name__)


def _extract_audio_from_file(video_path: Path, target_sr: int = 16000) -> Optional[np.ndarray]:
    """Extract audio track from an MP4 file or companion WAV as 16kHz mono float32."""
    # 1. Check for companion audio file with same name (.wav, .mp3)
    companion_audio = None
    for cand_ext in (".wav", ".mp3", ".m4a"):
        cand = video_path.with_suffix(cand_ext)
        if cand.is_file():
            companion_audio = cand
            break

    if companion_audio and companion_audio.is_file():
        try:
            data, sr = sf.read(str(companion_audio), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != target_sr:
                # Resample to target rate
                num_target_samples = int(len(data) * target_sr / sr)
                data = np.interp(
                    np.linspace(0.0, 1.0, num_target_samples, endpoint=False),
                    np.linspace(0.0, 1.0, len(data), endpoint=False),
                    data,
                ).astype(np.float32)
            logger.info("Loaded companion audio for %s: %d samples (%.1fs)",
                        video_path.name, len(data), len(data) / target_sr)
            return data
        except Exception as err:
            logger.warning("Could not read companion audio %s: %s", companion_audio, err)

    # 2. Extract embedded audio stream from video container via PyAV
    try:
        container = av.open(str(video_path))
        if not container.streams.audio:
            container.close()
            return None

        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)

        chunks = []
        for frame in container.decode(stream):
            frame.pts = None
            resampled = resampler.resample(frame)
            for rf in resampled:
                arr = rf.to_ndarray()
                chunks.append(arr.flatten())
        container.close()

        if chunks:
            raw_s16 = np.concatenate(chunks)
            float_samples = (raw_s16.astype(np.float32) / 32768.0).astype(np.float32)
            logger.info("Extracted embedded audio from %s: %d samples (%.1fs)",
                        video_path.name, len(float_samples), len(float_samples) / target_sr)
            return float_samples
    except Exception as err:
        logger.debug("No embedded audio stream in %s (%s)", video_path.name, err)

    return None


class ScenarioStream:
    """Manages simultaneous video decoding and synchronized audio streaming for one clip."""

    def __init__(
        self,
        video_path: Path,
        max_frame_size: int = 1080,
        sample_rate: int = 16000,
        window_seconds: float = 2.0,
        hop_seconds: float = 1.0,
        loop: bool = True,
    ):
        self.video_path = Path(video_path)
        self.max_frame_size = max_frame_size
        self.sample_rate = sample_rate
        self.window_samples = int(window_seconds * sample_rate)
        self.hop_samples = int(hop_seconds * sample_rate)
        self.loop = loop

        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0
        self.total_frames: int = 0
        self.frame_index: int = 0

        # Audio buffer
        self.audio_samples: Optional[np.ndarray] = None
        self.audio_ptr: int = 0
        self.window_index: int = 0
        self.last_audio_sec: float = -1.0

        self.is_opened: bool = False

    def open(self):
        """Open video capture and extract audio."""
        if not self.video_path.is_file():
            raise VideoSourceError(f"Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise VideoSourceError(f"Could not open video file: {self.video_path}")

        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(raw_fps) if (raw_fps and 0 < raw_fps <= 120) else 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.frame_index = 0

        # Extract audio
        self.audio_samples = _extract_audio_from_file(self.video_path, self.sample_rate)
        self.audio_ptr = 0
        self.window_index = 0
        self.last_audio_sec = -1.0
        self.is_opened = True

        logger.info("Opened ScenarioStream: %s (%.1f FPS, %d frames, audio=%s)",
                    self.video_path.name, self.fps, self.total_frames, self.has_audio)
        return self

    def close(self):
        """Release underlying video capture."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_opened = False

    @property
    def has_audio(self) -> bool:
        return self.audio_samples is not None and len(self.audio_samples) > 0

    def read(self, stride: int = 1) -> Tuple[Optional[VideoFrame], Optional[AudioWindow]]:
        """Read the next video frame and return any newly ready audio window."""
        if not self.is_opened or not self.cap:
            return None, None

        # Fast-forward intermediate frames if stride > 1
        if stride > 1:
            for _ in range(stride - 1):
                if not self.cap.grab():
                    if self.loop:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.frame_index = 0
                    break
                self.frame_index += 1

        success, img = self.cap.read()
        if not success:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_index = 0
                self.audio_ptr = 0
                success, img = self.cap.read()
            if not success:
                return None, None

        self.frame_index += 1
        current_time_sec = self.frame_index / self.fps

        # Resize video frame for responsive web streaming & inference
        h, w = img.shape[:2]
        longer = max(w, h)
        if longer > self.max_frame_size:
            scale = self.max_frame_size / float(longer)
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)

        video_frame = VideoFrame(image=img, index=self.frame_index, timestamp_sec=current_time_sec)

        # Synchronize audio window with video playback time
        audio_win = None
        if self.has_audio:
            # Check if 1 hop interval has elapsed since last window
            if current_time_sec - self.last_audio_sec >= (self.hop_samples / self.sample_rate):
                audio_win = self._next_audio_window(current_time_sec)
                self.last_audio_sec = current_time_sec

        return video_frame, audio_win

    def _next_audio_window(self, current_time_sec: float) -> Optional[AudioWindow]:
        """Slice the next window of audio samples."""
        if self.audio_samples is None:
            return None

        total_s = len(self.audio_samples)
        start = self.audio_ptr
        end = start + self.window_samples

        if end <= total_s:
            chunk = self.audio_samples[start:end]
            self.audio_ptr += self.hop_samples
        else:
            if self.loop:
                # Wrap around seamlessly
                part1 = self.audio_samples[start:total_s]
                needed = self.window_samples - len(part1)
                part2 = self.audio_samples[:needed]
                chunk = np.concatenate([part1, part2])
                self.audio_ptr = needed
            else:
                chunk = self.audio_samples[max(0, total_s - self.window_samples):total_s]

        # Calculate RMS dBFS
        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
        dbfs = float(20.0 * np.log10(max(rms, 1e-6)))
        peak = float(np.max(np.abs(chunk))) if len(chunk) > 0 else 0.0

        self.window_index += 1
        return AudioWindow(
            samples=chunk,
            sample_rate=self.sample_rate,
            index=self.window_index,
            start_time_sec=current_time_sec,
            rms_dbfs=dbfs,
            peak=peak,
        )

