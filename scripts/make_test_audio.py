"""
Generate SYNTHETIC test sounds into data/audio/synthetic/.

These are mathematically generated imitations (frequency-sweeping tones +
noise), NOT real recordings. They are used to test the audio plumbing
(Module 6) and the smoothing logic (Module 8). Real siren recordings are
still required to evaluate the siren classifier (Module 7).

    python scripts/make_test_audio.py

Files are written at 44.1 kHz on purpose, so the 44.1 -> 16 kHz resampling
path is exercised.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from app.utils.config import load_config  # noqa: E402

SAMPLE_RATE = 44100
rng = np.random.default_rng(seed=42)  # fixed seed -> identical files every run


def time_axis(seconds):
    return np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE


def siren(seconds, low_hz, high_hz, sweep_period_sec):
    """Tone whose pitch rises and falls between low_hz and high_hz (like a siren)."""
    t = time_axis(seconds)
    center, depth = (low_hz + high_hz) / 2, (high_hz - low_hz) / 2
    frequency = center + depth * np.sin(2 * np.pi * t / sweep_period_sec)
    phase = 2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    # Fundamental + two harmonics sounds more like a real siren than a pure sine
    tone = np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
    return 0.5 * tone / np.max(np.abs(tone))


def traffic_noise(seconds, level=0.15):
    """Low rumbling noise (white noise smoothed = more energy at low frequencies)."""
    noise = rng.normal(0, 1, len(time_axis(seconds)))
    rumble = np.convolve(noise, np.ones(40) / 40, mode="same")
    return level * rumble / np.max(np.abs(rumble))


def place(background, sound, start_sec):
    start = int(start_sec * SAMPLE_RATE)
    background[start:start + len(sound)] += sound[:len(background) - start]
    return background


def main():
    config = load_config()
    folder = Path(config["paths"]["audio_dir"]) / "synthetic"
    folder.mkdir(parents=True, exist_ok=True)

    clips = {
        # Continuous sirens over traffic noise
        "synthetic_siren_wail_10s.wav": traffic_noise(10) + siren(10, 600, 1500, 4.0),
        "synthetic_siren_yelp_10s.wav": traffic_noise(10) + siren(10, 700, 1600, 0.3),
        # No siren at all
        "synthetic_traffic_noise_10s.wav": traffic_noise(10),
        "synthetic_silence_5s.wav": np.zeros(len(time_axis(5))),
        # Siren from 3-8 s and 12-17 s (for Module 8: intermittent siren)
        "synthetic_intermittent_siren_20s.wav": place(place(
            traffic_noise(20), siren(5, 600, 1500, 4.0), 3), siren(5, 600, 1500, 4.0), 12),
        # A single 0.5 s siren-like blip (for Module 8: short false positive)
        "synthetic_short_blip_10s.wav": place(traffic_noise(10), siren(0.5, 700, 1600, 0.3), 5),
    }
    for name, audio in clips.items():
        audio = np.clip(audio, -1, 1).astype(np.float32)
        sf.write(folder / name, audio, SAMPLE_RATE)
        print(f"wrote {folder / name}  ({len(audio) / SAMPLE_RATE:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
