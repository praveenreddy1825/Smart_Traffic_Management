"""
Module 7 - Emergency siren classification with a pretrained Hugging Face model.

Model: MIT/ast-finetuned-audioset-10-10-0.4593 (Audio Spectrogram Transformer,
Gong, Chung & Glass, Interspeech 2021), fine-tuned on Google's AudioSet.
Its 527 labels really include: "Siren", "Civil defense siren", "Emergency vehicle",
"Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)".

    classifier = SirenClassifier.from_config(config)
    prediction = classifier.predict(audio_window)
    print(prediction.text())
        Detected sound: Emergency Siren
        Confidence: 0.91

How a window is classified:
  1. Feature extractor: 16 kHz audio -> 128-band log-mel spectrogram (1024 frames,
     ~10 s; shorter audio is padded), normalised with the model's own mean/std.
  2. AST forward pass -> 527 raw scores (logits).
  3. AudioSet is MULTI-LABEL (one clip can contain siren AND traffic), so each label
     gets its own SIGMOID probability (not softmax, where labels compete).
  4. Siren score = highest probability among the siren labels.
     score >= detection_threshold -> "Emergency Siren", otherwise "Other sound".
All numbers come straight from the model - nothing is invented.
"""

import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.audio.audio_input import resample
from app.utils.device import get_device
from app.utils.logger import get_logger

logger = get_logger(__name__)

SIREN = "Emergency Siren"
OTHER = "Other sound"


@dataclass
class SirenPrediction:
    is_siren: bool
    siren_score: float               # highest siren-label probability (0..1), from the model
    siren_type: str | None           # "Ambulance" / "Police" / "Fire truck" (only if siren)
    type_scores: dict                # probability of each siren type label
    top_labels: list                 # [(label, probability)] - 3 most likely AudioSet labels
    inference_ms: float              # model time for this window
    start_time_sec: float = 0.0      # where the window starts in the audio stream
    siren_label_scores: dict = field(default_factory=dict)

    @property
    def label(self):
        return SIREN if self.is_siren else OTHER

    def text(self):
        lines = [f"Detected sound: {self.label}", f"Confidence: {self.siren_score:.2f}"]
        if self.siren_type:
            lines.append(f"Type: {self.siren_type} ({self.type_scores[self.siren_type]:.2f})")
        return "\n".join(lines)


class SirenClassifier:
    def __init__(self, model_name, siren_labels, type_labels=None, detection_threshold=0.3,
                 device="auto", cache_dir=None):
        # Keep downloads inside the project (models/hf_cache) and quiet a Windows-only warning
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self._torch = torch
        self.model_name = model_name
        self.detection_threshold = detection_threshold
        self.device = get_device(device)

        logger.info("Loading audio model %s on %s", model_name, self.device)
        cache = str(cache_dir) if cache_dir else None
        # The model's own settings give one empty mel band at 16 kHz -> harmless warning
        warnings.filterwarnings("ignore", message="At least one mel filter has all zero values")
        try:
            # Edge-first: use the copy already in models/hf_cache, no internet needed
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_name, cache_dir=cache, local_files_only=True)
            self.model = AutoModelForAudioClassification.from_pretrained(
                model_name, cache_dir=cache, local_files_only=True)
        except OSError:
            logger.info("Model not in local cache - downloading once from Hugging Face")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir=cache)
            self.model = AutoModelForAudioClassification.from_pretrained(model_name, cache_dir=cache)
        self.model.to(self.device).eval()
        self.sample_rate = int(self.feature_extractor.sampling_rate)
        self.labels = self.model.config.id2label

        # Never assume labels exist - check them against the model's real label list
        label_to_id = self.model.config.label2id
        type_labels = dict(type_labels or {})
        missing = [name for name in list(siren_labels) + list(type_labels)
                   if name not in label_to_id]
        if missing:
            raise ValueError(f"Labels not found in {model_name}: {missing}")
        self.siren_label_ids = {name: label_to_id[name] for name in siren_labels}
        self.type_label_ids = {short: label_to_id[full] for full, short in type_labels.items()}
        logger.info("Siren labels (id): %s | threshold %.2f",
                    {name: idx for name, idx in self.siren_label_ids.items()},
                    detection_threshold)

    @classmethod
    def from_config(cls, config):
        settings = config["siren"]
        return cls(
            model_name=settings["model_name"],
            siren_labels=settings["siren_labels"],
            type_labels=settings.get("type_labels"),
            detection_threshold=settings.get("detection_threshold", 0.3),
            device=config.get("device", "auto"),
            cache_dir=Path(config["paths"]["hf_cache_dir"]) / "hub",
        )

    def probabilities(self, samples, sample_rate):
        """All 527 label probabilities (sigmoid) for one piece of mono audio."""
        samples = np.asarray(samples, dtype=np.float32)
        if sample_rate != self.sample_rate:
            samples = resample(samples, sample_rate, self.sample_rate)
        inputs = self.feature_extractor(samples, sampling_rate=self.sample_rate,
                                        return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            logits = self.model(**inputs).logits[0]
        return self._torch.sigmoid(logits).float().cpu().numpy()

    def predict(self, window=None, samples=None, sample_rate=None, start_time_sec=0.0):
        """Classify one AudioWindow (Module 6), or raw samples + sample_rate."""
        if window is not None:
            samples, sample_rate, start_time_sec = (window.samples, window.sample_rate,
                                                    window.start_time_sec)
        started = time.perf_counter()
        probabilities = self.probabilities(samples, sample_rate)
        if self.device == "cuda":
            self._torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000

        siren_label_scores = {name: float(probabilities[idx])
                              for name, idx in self.siren_label_ids.items()}
        siren_score = max(siren_label_scores.values())
        is_siren = siren_score >= self.detection_threshold
        type_scores = {short: float(probabilities[idx])
                       for short, idx in self.type_label_ids.items()}
        siren_type = max(type_scores, key=type_scores.get) if (is_siren and type_scores) else None
        top = np.argsort(probabilities)[::-1][:3]

        return SirenPrediction(
            is_siren=is_siren,
            siren_score=siren_score,
            siren_type=siren_type,
            type_scores=type_scores,
            top_labels=[(self.labels[int(i)], float(probabilities[i])) for i in top],
            inference_ms=inference_ms,
            start_time_sec=start_time_sec,
            siren_label_scores=siren_label_scores,
        )
