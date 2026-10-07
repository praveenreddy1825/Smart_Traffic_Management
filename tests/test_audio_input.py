"""Tests for Module 6: audio input (files generated in a temp folder; no microphone needed)."""

import time

import numpy as np
import pytest
import soundfile as sf

from app.audio.audio_input import (
    AudioInput, AudioSourceError, FileAudioInput, MicrophoneAudioInput,
    resample, rms_dbfs, to_mono)
from app.utils.config import load_config

TARGET_RATE = 16000


def write_tone(path, seconds, rate=44100, frequency=1000, channels=1, amplitude=0.5):
    t = np.arange(int(seconds * rate)) / rate
    tone = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    sf.write(path, data, rate)
    return path


def file_input(path, **overrides):
    settings = dict(sample_rate=TARGET_RATE, window_seconds=2.0, hop_seconds=1.0,
                    realtime_playback=False, loop_audio=False)
    settings.update(overrides)
    return FileAudioInput(path, **settings)


def dominant_frequency(samples, rate):
    spectrum = np.abs(np.fft.rfft(samples))
    return np.fft.rfftfreq(len(samples), 1 / rate)[np.argmax(spectrum)]


def test_resampling_keeps_pitch():
    rate = 44100
    tone = np.sin(2 * np.pi * 1000 * np.arange(rate) / rate).astype(np.float32)
    converted = resample(tone, rate, TARGET_RATE)
    assert len(converted) == TARGET_RATE
    assert dominant_frequency(converted, TARGET_RATE) == pytest.approx(1000, abs=2)


def test_stereo_becomes_mono():
    stereo = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    assert np.allclose(to_mono(stereo), [0.3, 0.7])


def test_loudness_in_dbfs():
    assert rms_dbfs(np.zeros(100)) == -120.0
    assert rms_dbfs(np.ones(100)) == pytest.approx(0.0)
    sine = np.sin(np.linspace(0, 200 * np.pi, 16000))
    assert rms_dbfs(sine) == pytest.approx(-3.0, abs=0.1)   # RMS of a full sine = 0.707


def test_overlapping_windows_from_file(tmp_path):
    path = write_tone(tmp_path / "tone.wav", seconds=5)
    windows = list(file_input(path).windows())
    # starts at 0,1,2,3 s give full 2 s windows (ending at 2..5 s)
    assert [w.start_time_sec for w in windows] == [0.0, 1.0, 2.0, 3.0]
    assert all(len(w.samples) == 2 * TARGET_RATE for w in windows)
    assert all(w.sample_rate == TARGET_RATE and w.samples.dtype == np.float32 for w in windows)
    assert not any(w.padded for w in windows)
    assert dominant_frequency(windows[0].samples, TARGET_RATE) == pytest.approx(1000, abs=2)


def test_tail_shorter_than_hop_is_padded(tmp_path):
    path = write_tone(tmp_path / "tone.wav", seconds=4.5)
    windows = list(file_input(path).windows())
    assert [w.start_time_sec for w in windows] == [0.0, 1.0, 2.0, 3.0]
    assert windows[-1].padded           # 3.0-5.0 s window, audio ends at 4.5 s


def test_clip_shorter_than_a_window_is_padded(tmp_path):
    path = write_tone(tmp_path / "short.wav", seconds=0.5)
    windows = list(file_input(path).windows())
    assert len(windows) == 1 and windows[0].padded
    assert len(windows[0].samples) == 2 * TARGET_RATE


def test_stereo_file_and_silence(tmp_path):
    path = write_tone(tmp_path / "stereo.wav", seconds=3, channels=2)
    assert len(list(file_input(path).windows())) == 2
    silent = tmp_path / "silence.wav"
    sf.write(silent, np.zeros(3 * 44100, dtype=np.float32), 44100)
    windows = list(file_input(silent).windows())
    assert windows and all(w.is_quiet(-50) for w in windows)   # silence still yields windows


def test_looping_continues_past_the_end(tmp_path):
    path = write_tone(tmp_path / "tone.wav", seconds=3)
    generator = file_input(path, loop_audio=True).windows()
    windows = [next(generator) for _ in range(6)]
    assert [w.start_time_sec for w in windows] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert all(len(w.samples) == 2 * TARGET_RATE for w in windows)


def test_realtime_playback_paces_like_a_microphone(tmp_path):
    path = write_tone(tmp_path / "tone.wav", seconds=3, rate=TARGET_RATE)
    source = file_input(path, realtime_playback=True, window_seconds=1.0, hop_seconds=0.5)
    started = time.perf_counter()
    windows = list(source.windows())
    elapsed = time.perf_counter() - started
    # last full window ends at 3.0 s, so it cannot arrive before ~3 s of wall time
    assert len(windows) == 5
    assert elapsed >= 2.9


def test_missing_and_invalid_files_raise(tmp_path):
    with pytest.raises(AudioSourceError, match="not found"):
        file_input(tmp_path / "nothing.wav").open()
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not really audio")
    with pytest.raises(AudioSourceError, match="Could not read"):
        file_input(broken).open()


def test_invalid_window_settings_rejected():
    with pytest.raises(ValueError):
        AudioInput(window_seconds=1.0, hop_seconds=2.0)


def test_from_config_chooses_source_type(tmp_path):
    config = load_config()
    assert isinstance(AudioInput.from_config(config, source="mic"), MicrophoneAudioInput)
    from_file = AudioInput.from_config(config, source=str(tmp_path / "a.wav"))
    assert isinstance(from_file, FileAudioInput)
    assert from_file.sample_rate == config["audio"]["sample_rate"]
