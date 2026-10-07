"""
Module 6 - Audio Input.

Delivers audio from the live microphone OR a recorded file as a stream of
short, overlapping windows - the input format for the siren classifier:

    with AudioInput.from_config(config) as audio:        # source: "mic" or a file path
        for window in audio.windows():
            window.samples        # float32 mono, values -1..1, at window.sample_rate (16 kHz)
            window.start_time_sec # where the window starts in the stream
            window.rms_dbfs       # loudness (for "mic active / quiet" display)

Preprocessing applied to every source:
    multi-channel -> mono (average of channels)
    any sample rate -> target rate (default 16 kHz, polyphase resampling)
    float32 in the range -1..1
"""

import queue
import time
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from app.utils.config import resolve_path
from app.utils.logger import get_logger

logger = get_logger(__name__)

MIC_SOURCE_NAMES = {"mic", "microphone"}


class AudioSourceError(Exception):
    """Raised when an audio file or microphone cannot be opened/read."""


@dataclass
class AudioWindow:
    samples: np.ndarray     # float32 mono samples, -1..1
    sample_rate: int        # samples per second (e.g. 16000)
    index: int              # window number: 0, 1, 2, ...
    start_time_sec: float   # start of this window in the audio stream
    rms_dbfs: float         # loudness in dB relative to full scale (0 = max, -inf = silence)
    peak: float             # largest absolute sample value (0..1)
    padded: bool = False    # True if silence was added at the end (clip shorter than a window)

    @property
    def duration_sec(self):
        return len(self.samples) / self.sample_rate

    def is_quiet(self, threshold_dbfs=-50.0):
        return self.rms_dbfs < threshold_dbfs


# ---------------------------------------------------------------- preprocessing helpers

def to_mono(audio):
    """(samples, channels) -> (samples,) by averaging channels."""
    audio = np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def resample(audio, original_rate, target_rate):
    """Change the sample rate without changing pitch (polyphase filtering)."""
    original_rate, target_rate = int(original_rate), int(target_rate)
    if original_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    divisor = gcd(original_rate, target_rate)
    return resample_poly(audio, target_rate // divisor, original_rate // divisor).astype(np.float32)


def rms_dbfs(samples):
    """Loudness of a window in dBFS (decibels relative to full scale). Silence -> -120."""
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if len(samples) else 0.0
    return max(20 * np.log10(rms), -120.0) if rms > 0 else -120.0


def make_window(samples, sample_rate, index, start_time_sec, padded=False):
    samples = np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)
    return AudioWindow(samples=samples, sample_rate=sample_rate, index=index,
                       start_time_sec=start_time_sec, rms_dbfs=rms_dbfs(samples),
                       peak=float(np.max(np.abs(samples))) if len(samples) else 0.0,
                       padded=padded)


# ---------------------------------------------------------------- sources

class AudioInput:
    """Common settings + factory. Use FileAudioInput or MicrophoneAudioInput."""

    def __init__(self, sample_rate=16000, window_seconds=2.0, hop_seconds=1.0):
        if hop_seconds <= 0 or window_seconds <= 0 or hop_seconds > window_seconds:
            raise ValueError("audio: need 0 < hop_seconds <= window_seconds")
        self.sample_rate = int(sample_rate)
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.window_size = int(round(window_seconds * self.sample_rate))
        self.hop_size = int(round(hop_seconds * self.sample_rate))

    @staticmethod
    def from_config(config, source=None):
        settings = config["audio"]
        source = settings["source"] if source is None else source
        common = dict(sample_rate=settings.get("sample_rate", 16000),
                      window_seconds=settings.get("window_seconds", 2.0),
                      hop_seconds=settings.get("hop_seconds", 1.0))
        if str(source).strip().lower() in MIC_SOURCE_NAMES:
            return MicrophoneAudioInput(device=settings.get("microphone_device"), **common)
        return FileAudioInput(resolve_path(str(source)),
                              realtime_playback=settings.get("realtime_playback", True),
                              loop_audio=settings.get("loop_audio", False), **common)

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc_info):
        self.close()

    def open(self):
        return self

    def close(self):
        pass


class FileAudioInput(AudioInput):
    """Recorded audio file. The whole file is loaded once (test clips are small)."""

    def __init__(self, path, realtime_playback=True, loop_audio=False, **settings):
        super().__init__(**settings)
        self.path = Path(path)
        self.realtime_playback = realtime_playback
        self.loop_audio = loop_audio
        self.audio = None
        self.original_sample_rate = None
        self.description = f"file {self.path.name}"

    @property
    def duration_sec(self):
        return 0.0 if self.audio is None else len(self.audio) / self.sample_rate

    def open(self):
        import soundfile as sf

        if not self.path.is_file():
            raise AudioSourceError(f"Audio file not found: {self.path}")
        try:
            data, self.original_sample_rate = sf.read(str(self.path), dtype="float32",
                                                      always_2d=True)
        except Exception as error:  # unsupported/corrupt file
            raise AudioSourceError(f"Could not read audio file {self.path}: {error}") from error
        if len(data) == 0:
            raise AudioSourceError(f"Audio file is empty: {self.path}")

        self.audio = resample(to_mono(data), self.original_sample_rate, self.sample_rate)
        logger.info("Opened %s | %d Hz, %d ch -> %d Hz mono | %.1f s",
                    self.description, self.original_sample_rate, data.shape[1],
                    self.sample_rate, self.duration_sec)
        return self

    def windows(self):
        """Yield overlapping windows. In real-time mode a window is released when its
        END time is reached - exactly when a microphone would have delivered it."""
        if self.audio is None:
            self.open()
        total = len(self.audio)
        start_clock = time.perf_counter()
        start, index = 0, 0

        while True:
            end = start + self.window_size
            padded = False
            if self.loop_audio:
                # Wrap around the end of the clip so looping never leaves a gap
                samples = np.take(self.audio, np.arange(start, end), mode="wrap")
            elif end <= total:
                samples = self.audio[start:end]
            elif index == 0 or (start - self.hop_size) + self.window_size < total:
                # Last bit of audio not covered by a full window: pad it with silence
                samples = np.zeros(self.window_size, dtype=np.float32)
                samples[:total - start] = self.audio[start:total]
                padded = True
            else:
                return

            if self.realtime_playback:
                delay = start_clock + end / self.sample_rate - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

            yield make_window(samples, self.sample_rate, index, start / self.sample_rate, padded)
            if padded:
                return
            start += self.hop_size
            index += 1


class MicrophoneAudioInput(AudioInput):
    """Live microphone via sounddevice. Recording runs in the background (callback),
    so no audio is lost while the main program is busy."""

    def __init__(self, device=None, no_audio_timeout_sec=5.0, **settings):
        super().__init__(**settings)
        self.device = device
        self.no_audio_timeout_sec = no_audio_timeout_sec
        self.capture_rate = None       # rate the microphone actually records at
        self.overflow_count = 0        # times the background buffer overflowed
        self.description = "microphone"
        self._stream = None
        self._chunks = queue.Queue()

    def _callback(self, indata, frames, time_info, status):
        """Called by sounddevice in a background thread for every small block of audio."""
        if status.input_overflow:
            self.overflow_count += 1
        self._chunks.put(indata.copy())

    def open(self):
        import sounddevice as sd

        try:
            device_info = sd.query_devices(self.device, "input")
        except Exception as error:
            raise AudioSourceError(f"No usable microphone ({self.device or 'default'}): "
                                   f"{error}") from error
        self.description = f"microphone '{device_info['name']}'"

        # Try to record directly at the target rate; if the device refuses,
        # record at its own default rate and resample each window.
        for rate in (self.sample_rate, int(device_info["default_samplerate"])):
            try:
                stream = sd.InputStream(samplerate=rate, channels=1, dtype="float32",
                                        device=self.device, callback=self._callback)
                stream.start()
            except Exception as error:
                logger.warning("Microphone could not open at %d Hz: %s", rate, error)
                continue
            self._stream, self.capture_rate = stream, rate
            logger.info("Opened %s | recording at %d Hz -> %d Hz mono windows",
                        self.description, rate, self.sample_rate)
            return self
        raise AudioSourceError(f"Could not open {self.description}")

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Microphone closed (buffer overflows: %d)", self.overflow_count)

    def windows(self):
        """Yield a window every hop_seconds, forever (until the caller stops)."""
        if self._stream is None:
            self.open()
        window_size = int(round(self.window_seconds * self.capture_rate))
        hop_size = int(round(self.hop_seconds * self.capture_rate))
        buffer = np.zeros(0, dtype=np.float32)
        consumed_samples, index = 0, 0
        reported_overflows = 0

        try:
            while True:
                try:
                    chunk = self._chunks.get(timeout=self.no_audio_timeout_sec)
                except queue.Empty:
                    raise AudioSourceError(
                        f"No audio received from the microphone for "
                        f"{self.no_audio_timeout_sec:.0f} s (unplugged or blocked?)") from None
                buffer = np.concatenate([buffer, to_mono(chunk)])

                if self.overflow_count > reported_overflows:
                    logger.warning("Microphone buffer overflow (processing too slow) - "
                                   "some audio was dropped; continuing")
                    reported_overflows = self.overflow_count

                while len(buffer) >= window_size:
                    samples = resample(buffer[:window_size], self.capture_rate, self.sample_rate)
                    yield make_window(samples, self.sample_rate, index,
                                      consumed_samples / self.capture_rate)
                    buffer = buffer[hop_size:]
                    consumed_samples += hop_size
                    index += 1
        finally:
            self.close()


def list_input_devices():
    """Return [(index, name, channels, default_rate)] of all microphones."""
    import sounddevice as sd
    return [(index, device["name"], device["max_input_channels"], device["default_samplerate"])
            for index, device in enumerate(sd.query_devices())
            if device["max_input_channels"] > 0]
