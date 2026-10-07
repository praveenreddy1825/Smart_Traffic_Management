"""Tests for Module 7: SirenClassifier.

Uses the real AST model from models/hf_cache. The ESC-50 tests run only if
scripts/download_esc50_subset.py has been run (real siren recordings).
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from app.audio.audio_input import FileAudioInput
from app.audio.siren_classifier import OTHER, SIREN, SirenClassifier, SirenPrediction
from app.utils.config import load_config

CONFIG = load_config()
ESC50 = Path(CONFIG["paths"]["audio_dir"]) / "esc50"


@pytest.fixture(scope="module")
def classifier():
    return SirenClassifier.from_config(CONFIG)


def esc50_clip(category):
    labels = ESC50 / "labels.csv"
    if not labels.exists():
        pytest.skip("ESC-50 subset not downloaded (scripts/download_esc50_subset.py)")
    row = next(r for r in csv.DictReader(open(labels, encoding="utf-8"))
               if r["category"] == category and r["fold"] == "3")   # test fold
    return ESC50 / row["filename"]


def whole_clip(path):
    source = FileAudioInput(path, realtime_playback=False, sample_rate=16000,
                            window_seconds=2.0, hop_seconds=1.0).open()
    return source.audio


def test_model_really_has_the_siren_labels(classifier):
    labels = set(classifier.labels.values())
    for name in ["Siren", "Ambulance (siren)", "Police car (siren)",
                 "Fire engine, fire truck (siren)", "Emergency vehicle"]:
        assert name in labels
    assert classifier.sample_rate == 16000


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError, match="not found"):
        SirenClassifier(CONFIG["siren"]["model_name"], siren_labels=["Siren", "Ambulance horn"],
                        cache_dir=Path(CONFIG["paths"]["hf_cache_dir"]) / "hub")


def test_silence_is_not_a_siren(classifier):
    prediction = classifier.predict(samples=np.zeros(32000, dtype=np.float32), sample_rate=16000)
    assert not prediction.is_siren
    assert prediction.label == OTHER and prediction.siren_type is None


def test_probabilities_are_independent_sigmoids(classifier):
    probabilities = classifier.probabilities(np.zeros(32000, dtype=np.float32), 16000)
    assert probabilities.shape == (527,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert not np.isclose(probabilities.sum(), 1.0)   # multi-label: not a softmax


def test_other_sample_rates_are_resampled(classifier):
    tone = np.sin(2 * np.pi * 440 * np.arange(44100 * 2) / 44100).astype(np.float32)
    prediction = classifier.predict(samples=tone, sample_rate=44100)
    assert 0.0 <= prediction.siren_score <= 1.0


def test_real_siren_recording_detected(classifier):
    prediction = classifier.predict(samples=whole_clip(esc50_clip("siren")), sample_rate=16000)
    assert prediction.is_siren and prediction.label == SIREN
    assert prediction.siren_type in {"Ambulance", "Police", "Fire truck"}


@pytest.mark.parametrize("category", ["car_horn", "clock_alarm", "engine"])
def test_real_non_siren_recordings_rejected(classifier, category):
    prediction = classifier.predict(samples=whole_clip(esc50_clip(category)), sample_rate=16000)
    assert not prediction.is_siren


def test_prediction_text_format():
    prediction = SirenPrediction(is_siren=True, siren_score=0.91, siren_type="Ambulance",
                                 type_scores={"Ambulance": 0.7}, top_labels=[], inference_ms=1.0)
    assert prediction.text() == ("Detected sound: Emergency Siren\nConfidence: 0.91\n"
                                 "Type: Ambulance (0.70)")
