"""
Build the four Module 8 test scenarios from REAL ESC-50 recordings.

Background = real street-type clips (engine, car horn, train, rain, ...).
Siren      = real ESC-50 siren clips, mixed ON TOP of the background.
Only ESC-50 folds 3-5 are used (folds 1-2 were used to choose the siren threshold).

Writes data/audio/scenarios/*.wav (16 kHz mono) + scenarios.json with the exact
siren intervals (ground truth) and the expected behaviour.

    python scripts/download_esc50_subset.py      # once
    python scripts/make_smoothing_scenarios.py
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from app.audio.audio_input import FileAudioInput, rms_dbfs  # noqa: E402
from app.utils.config import load_config  # noqa: E402

RATE = 16000
SCENARIO_FOLDS = {"3", "4", "5"}
BACKGROUND_CLASSES = {"engine", "car_horn", "train", "airplane", "helicopter",
                      "footsteps", "rain", "wind"}
BACKGROUND_GAIN = 0.6   # background a bit quieter than the siren, like a nearby ambulance

SCENARIOS = [
    {"name": "1_background_noise", "seconds": 60, "sirens": [],
     "expected": "never active"},
    {"name": "2_short_false_positive", "seconds": 40, "sirens": [[20.0, 21.5]],
     "expected": "never active (1.5 s burst is too short to confirm)"},
    {"name": "3_continuous_siren", "seconds": 45, "sirens": [[10.0, 30.0]],
     "expected": "one event, starts a few s after 10 s, ends a few s after 30 s"},
    {"name": "4_intermittent_siren", "seconds": 50, "sirens": [[10.0, 20.0], [23.0, 33.0]],
     "expected": "ONE event covering both parts (3 s gap bridged)"},
]


def load(path):
    source = FileAudioInput(path, realtime_playback=False, sample_rate=RATE,
                            window_seconds=1.0, hop_seconds=1.0).open()
    return source.audio


def mostly_sound(audio, chunk_sec=0.5, min_dbfs=-45.0, min_share=0.9):
    """True if >= 90% of 0.5 s chunks are not near-silent (skips clips padded with silence)."""
    size = int(chunk_sec * RATE)
    chunks = [audio[i:i + size] for i in range(0, len(audio) - size + 1, size)]
    return np.mean([rms_dbfs(chunk) > min_dbfs for chunk in chunks]) >= min_share


def build_track(clips, seconds):
    """Concatenate clips (repeating if needed) to exactly `seconds` long."""
    track, i = [], 0
    while sum(len(c) for c in track) < seconds * RATE:
        track.append(clips[i % len(clips)])
        i += 1
    return np.concatenate(track)[:int(seconds * RATE)]


def fade(audio, fade_sec=0.05):
    ramp = np.linspace(0, 1, int(fade_sec * RATE))
    audio = audio.copy()
    audio[:len(ramp)] *= ramp
    audio[-len(ramp):] *= ramp[::-1]
    return audio


def main():
    config = load_config()
    esc50 = Path(config["paths"]["audio_dir"]) / "esc50"
    rows = [r for r in csv.DictReader(open(esc50 / "labels.csv", encoding="utf-8"))
            if r["fold"] in SCENARIO_FOLDS]
    background_clips = [load(esc50 / r["filename"]) for r in rows
                        if r["category"] in BACKGROUND_CLASSES]
    siren_clips = [audio for audio in (load(esc50 / r["filename"]) for r in rows
                                       if r["category"] == "siren") if mostly_sound(audio)]
    print(f"Using {len(background_clips)} background clips and {len(siren_clips)} "
          f"siren clips (siren clips that are mostly silence were skipped)")

    folder = Path(config["paths"]["audio_dir"]) / "scenarios"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = []
    for scenario_number, scenario in enumerate(SCENARIOS):
        rng = np.random.default_rng(scenario_number)
        background = build_track([background_clips[i] for i in
                                  rng.permutation(len(background_clips))], scenario["seconds"])
        mix = BACKGROUND_GAIN * background
        for start, end in scenario["sirens"]:
            siren = fade(build_track([siren_clips[i] for i in
                                      rng.permutation(len(siren_clips))], end - start))
            mix[int(start * RATE):int(start * RATE) + len(siren)] += siren
        mix = (mix / max(1.0, np.max(np.abs(mix)))).astype(np.float32)   # avoid clipping
        path = folder / f"{scenario['name']}.wav"
        sf.write(path, mix, RATE)
        manifest.append({**scenario, "file": path.name})
        print(f"wrote {path.name}: {scenario['seconds']} s, siren intervals {scenario['sirens']}")

    (folder / "scenarios.json").write_text(json.dumps(manifest, indent=2))
    print(f"Ground truth saved to {folder / 'scenarios.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
