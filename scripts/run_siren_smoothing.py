"""
Module 8 demo: raw per-window siren decisions vs the smoothed emergency state.

Live / single file:
    python scripts/run_siren_smoothing.py --source mic
    python scripts/run_siren_smoothing.py --source data/audio/scenarios/3_continuous_siren.wav

Evaluate the four test scenarios against their ground truth:
    python scripts/make_smoothing_scenarios.py     # once
    python scripts/run_siren_smoothing.py --evaluate

Evaluation reports, per scenario: confirmed events, false activations,
activation delay (siren start -> emergency confirmed) and release delay
(siren end -> emergency ended). Saved to data/test/output/smoothing_eval.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio.audio_input import AudioInput, AudioSourceError, FileAudioInput  # noqa: E402
from app.audio.siren_classifier import SirenClassifier  # noqa: E402
from app.audio.siren_smoother import SirenSmoother  # noqa: E402
from app.utils.config import load_config  # noqa: E402
from app.utils.logger import get_logger, setup_logging  # noqa: E402


def process(audio, classifier, smoother, verbose=True, max_seconds=None):
    """Run classifier + smoother over a source; returns (raw siren flags, list of events).
    An event is [confirmed_at_sec, ended_at_sec or None]."""
    events, raw_flags = [], []
    was_active = False
    if verbose:
        print(f"\n{'win':>4} {'end':>6}  {'raw':<6} {'score':>5}  {'history':>7}  "
              f"{'state':<16} emergency")
    for window in audio.windows():
        prediction = classifier.predict(window)
        state = smoother.update(prediction)
        raw_flags.append(prediction.is_siren)
        if state.emergency_active and not was_active:
            events.append([state.timestamp_sec, None])
        if was_active and not state.emergency_active:
            events[-1][1] = state.timestamp_sec
        was_active = state.emergency_active
        if verbose:
            print(f"{window.index:>4} {state.timestamp_sec:>5.1f}s  "
                  f"{'SIREN' if prediction.is_siren else '-':<6} {prediction.siren_score:>5.2f}  "
                  f"{state.siren_windows_in_history}/{state.history_length:<5}  "
                  f"{state.state:<16} {'ACTIVE' if state.emergency_active else '-'}"
                  f"{'  (' + state.siren_type + ')' if state.siren_type else ''}")
        if max_seconds and state.timestamp_sec >= max_seconds:
            break
    return raw_flags, events


def evaluate(config, classifier):
    folder = Path(config["paths"]["audio_dir"]) / "scenarios"
    manifest_file = folder / "scenarios.json"
    if not manifest_file.exists():
        print("Run scripts/make_smoothing_scenarios.py first.")
        return 1
    audio_settings = config["audio"]
    results = []
    for scenario in json.loads(manifest_file.read_text()):
        audio = FileAudioInput(folder / scenario["file"], realtime_playback=False,
                               sample_rate=classifier.sample_rate,
                               window_seconds=audio_settings["window_seconds"],
                               hop_seconds=audio_settings["hop_seconds"])
        smoother = SirenSmoother.from_config(config)
        raw_flags, events = process(audio, classifier, smoother, verbose=False)
        sirens = scenario["sirens"]

        def overlaps(event):
            end = event[1] if event[1] is not None else float("inf")
            return any(event[0] <= s_end + 10 and end >= s_start for s_start, s_end in sirens)

        false_activations = sum(not overlaps(event) for event in events)
        result = {
            "scenario": scenario["name"], "expected": scenario["expected"],
            "siren_intervals": sirens, "raw_siren_windows": int(sum(raw_flags)),
            "total_windows": len(raw_flags), "events": events,
            "false_activations": false_activations,
        }
        if sirens and events:
            result["activation_delay_sec"] = round(events[0][0] - sirens[0][0], 1)
            if events[-1][1] is not None:
                result["release_delay_sec"] = round(events[-1][1] - sirens[-1][1], 1)
        results.append(result)

    print(f"\n{'scenario':<26} {'raw siren win':>13} {'events':>6} {'false act.':>10} "
          f"{'activation delay':>17} {'release delay':>14}")
    for r in results:
        print(f"{r['scenario']:<26} {r['raw_siren_windows']:>6}/{r['total_windows']:<6} "
              f"{len(r['events']):>6} {r['false_activations']:>10} "
              f"{str(r.get('activation_delay_sec', '-')) + ' s':>17} "
              f"{str(r.get('release_delay_sec', '-')) + ' s':>14}")
        print(f"{'':<26} events (confirmed -> ended, s): {r['events']} | expected: {r['expected']}")

    output = Path(config["paths"]["test_data_dir"]) / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoothing_eval.json").write_text(json.dumps({
        "smoothing": config["smoothing"], "threshold": classifier.detection_threshold,
        "window_seconds": audio_settings["window_seconds"],
        "hop_seconds": audio_settings["hop_seconds"], "results": results}, indent=2))
    print(f"\nSaved {output / 'smoothing_eval.json'}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Siren temporal smoothing (Module 8)")
    parser.add_argument("--source", help='"mic" or an audio file (default: config)')
    parser.add_argument("--evaluate", action="store_true", help="score the 4 test scenarios")
    parser.add_argument("--no-realtime", action="store_true", help="read files as fast as possible")
    parser.add_argument("--seconds", type=float, help="stop after this much audio")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)
    logger = get_logger("run_siren_smoothing")
    classifier = SirenClassifier.from_config(config)
    if args.evaluate:
        return evaluate(config, classifier)

    audio = AudioInput.from_config(config, source=args.source)
    if args.no_realtime and hasattr(audio, "realtime_playback"):
        audio.realtime_playback = False
    try:
        with audio:
            _, events = process(audio, classifier, SirenSmoother.from_config(config),
                                max_seconds=args.seconds)
    except AudioSourceError as error:
        logger.error("%s", error)
        return 1
    except KeyboardInterrupt:
        logger.info("Stopped with Ctrl+C")
        return 0
    print(f"\nConfirmed emergency events (confirmed -> ended, s): {events or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
