"""
Module 7 demo: classify each audio window as Emergency Siren / Other sound.

Examples:
    python scripts/run_siren_detection.py --source mic
    python scripts/run_siren_detection.py --source data/audio/esc50/1-172649-A-40.wav
    python scripts/run_siren_detection.py --source data/audio/synthetic/synthetic_intermittent_siren_20s.wav

NOTE: this shows the RAW decision per window. A single window does not yet
trigger an emergency - temporal smoothing is added in Module 8.
Press Ctrl+C to stop the microphone.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio.audio_input import AudioInput, AudioSourceError  # noqa: E402
from app.audio.siren_classifier import SirenClassifier  # noqa: E402
from app.utils.config import load_config  # noqa: E402
from app.utils.logger import get_logger, setup_logging  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Siren detection (Module 7)")
    parser.add_argument("--source", help='"mic" or an audio file (default: config)')
    parser.add_argument("--seconds", type=float, help="stop after this much audio")
    parser.add_argument("--no-realtime", action="store_true", help="read files as fast as possible")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)
    logger = get_logger("run_siren_detection")
    if args.device:
        config["device"] = args.device

    classifier = SirenClassifier.from_config(config)
    audio = AudioInput.from_config(config, source=args.source)
    if args.no_realtime and hasattr(audio, "realtime_playback"):
        audio.realtime_playback = False
    if audio.sample_rate != classifier.sample_rate:
        logger.warning("audio.sample_rate (%d) differs from the model's %d Hz - windows are "
                       "resampled before classification", audio.sample_rate, classifier.sample_rate)

    siren_windows = total_windows = 0
    try:
        with audio:
            print(f"\n{'win':>4} {'start':>7} {'dBFS':>6}  {'result':<16} {'conf':>5}  "
                  f"{'type':<16} top AudioSet labels")
            for window in audio.windows():
                prediction = classifier.predict(window)
                total_windows += 1
                siren_windows += prediction.is_siren
                siren_type = (f"{prediction.siren_type} {prediction.type_scores[prediction.siren_type]:.2f}"
                              if prediction.siren_type else "-")
                top = ", ".join(f"{label[:20]} {score:.2f}" for label, score in prediction.top_labels)
                print(f"{window.index:>4} {window.start_time_sec:>6.1f}s {window.rms_dbfs:>6.1f}  "
                      f"{prediction.label:<16} {prediction.siren_score:>5.2f}  {siren_type:<16} {top}")
                if args.seconds and window.start_time_sec + window.duration_sec >= args.seconds:
                    break
    except AudioSourceError as error:
        logger.error("%s", error)
        return 1
    except KeyboardInterrupt:
        logger.info("Stopped with Ctrl+C")

    print(f"\nWindows classified as siren: {siren_windows}/{total_windows} "
          f"(threshold {classifier.detection_threshold})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
