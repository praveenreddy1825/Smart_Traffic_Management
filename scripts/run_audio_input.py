"""
Module 6 demo: read audio from the microphone or a file and show each window's loudness.

Examples:
    python scripts/run_audio_input.py --list-devices
    python scripts/run_audio_input.py --source mic --seconds 10
    python scripts/run_audio_input.py --source data/audio/synthetic/synthetic_siren_wail_10s.wav
    python scripts/run_audio_input.py --source mic --record data/audio/my_recording.wav --seconds 15

Press Ctrl+C to stop the microphone.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.audio.audio_input import (  # noqa: E402
    AudioInput, AudioSourceError, list_input_devices)
from app.utils.config import load_config, resolve_path  # noqa: E402
from app.utils.logger import get_logger, setup_logging  # noqa: E402


def level_bar(dbfs, width=30):
    """Text loudness meter: -60 dBFS = empty, 0 dBFS = full."""
    filled = int(np.clip((dbfs + 60) / 60, 0, 1) * width)
    return "#" * filled + "." * (width - filled)


def main():
    parser = argparse.ArgumentParser(description="Audio input test (Module 6)")
    parser.add_argument("--source", help='"mic" or an audio file (default: config)')
    parser.add_argument("--seconds", type=float, help="stop after this much audio")
    parser.add_argument("--no-realtime", action="store_true", help="read files as fast as possible")
    parser.add_argument("--record", help="save the received audio (non-overlapping) to this WAV")
    parser.add_argument("--list-devices", action="store_true", help="list microphones and exit")
    args = parser.parse_args()

    if args.list_devices:
        for index, name, channels, rate in list_input_devices():
            print(f"[{index:>2}] {name}  ({channels} ch, default {rate:.0f} Hz)")
        return 0

    config = load_config()
    setup_logging(config)
    logger = get_logger("run_audio_input")
    quiet_threshold = config["audio"].get("silence_threshold_dbfs", -50.0)

    audio = AudioInput.from_config(config, source=args.source)
    if args.no_realtime and hasattr(audio, "realtime_playback"):
        audio.realtime_playback = False

    recorded = []
    windows_seen = 0
    start = time.perf_counter()
    try:
        with audio:
            print(f"\n{'win':>4} {'start':>7}  {'dBFS':>6}  {'peak':>5}  level")
            for window in audio.windows():
                windows_seen += 1
                state = "quiet" if window.is_quiet(quiet_threshold) else "sound"
                print(f"{window.index:>4} {window.start_time_sec:>6.1f}s  {window.rms_dbfs:>6.1f}  "
                      f"{window.peak:>5.2f}  {level_bar(window.rms_dbfs)} {state}"
                      f"{'  (padded)' if window.padded else ''}")
                # Keep only the NEW part of each window (the first hop) to avoid duplicates
                recorded.append(window.samples if window.index == 0
                                else window.samples[-audio.hop_size:])
                if args.seconds and window.start_time_sec + window.duration_sec >= args.seconds:
                    break
    except AudioSourceError as error:
        logger.error("%s", error)
        return 1
    except KeyboardInterrupt:
        logger.info("Stopped with Ctrl+C")

    elapsed = time.perf_counter() - start
    print(f"\nWindows: {windows_seen}  |  window {audio.window_seconds}s, hop {audio.hop_seconds}s, "
          f"{audio.sample_rate} Hz  |  wall time {elapsed:.1f}s")

    if args.record and recorded:
        import soundfile as sf
        path = resolve_path(args.record)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.concatenate(recorded), audio.sample_rate)
        print(f"Saved recording to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
