"""
SmartCity Sentinel - Main Pipeline Entrypoint (Module 12).

Integrates multi-directional traffic video feeds with audio siren classification
and directional emergency fusion.

Run in console mode:
    python main.py
    python main.py --audio data/audio/scenarios/3_continuous_siren.wav

Run interactive Web dashboard:
    python main.py --dashboard
    python app.py
    python scripts/run_dashboard.py
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure the script runs using the project's virtual environment (.venv)
ROOT_DIR = Path(__file__).resolve().parent
venv_python = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
if venv_python.is_file() and sys.executable.lower() != str(venv_python).lower():
    sys.exit(subprocess.call([str(venv_python)] + sys.argv))

from app.audio.audio_input import AudioInput
from app.audio.siren_classifier import SirenClassifier
from app.audio.siren_smoother import AudioEmergencyState, SirenSmoother
from app.decision.directional_fusion import DirectionalFusion
from app.traffic.multi_approach_manager import (
    APPROACH_DIRECTIONS, MultiApproachManager)
from app.utils.config import ensure_directories, load_config
from app.utils.logger import get_logger, setup_logging


def run_pipeline(config, audio_source=None, max_seconds=None):
    logger = get_logger("main")
    logger.info("Starting Multi-Directional Traffic & Emergency Monitoring Pipeline...")

    manager = MultiApproachManager.from_config(config, loop_videos=True)
    classifier = SirenClassifier.from_config(config)
    smoother = SirenSmoother.from_config(config)
    fusion = DirectionalFusion()

    audio_src = audio_source or config["audio"]["source"]
    audio_input = None
    try:
        audio_input = AudioInput.from_config(config, source=audio_src)
        audio_input.open()
        logger.info("Audio input active: %s", audio_src)
    except Exception as err:
        logger.warning("Could not open audio input %s: %s", audio_src, err)

    audio_windows_gen = audio_input.windows() if audio_input else None
    latest_audio_state = AudioEmergencyState(
        state="NO_SIREN", emergency_active=False, window_is_siren=False,
        window_score=0.0, siren_windows_in_history=0, history_length=5,
        smoothed_confidence=0.0, siren_type=None, event_started_sec=None,
        event_duration_sec=0.0, timestamp_sec=0.0, events_so_far=0,
    )

    manager.open()
    logger.info("All 4 directional approach video feeds opened.")
    print("\n" + "=" * 90)
    print(" SmartCity Sentinel - Live 4-Way Traffic & Directional Emergency Monitor")
    print("=" * 90)
    print(f"{'Time':>6} | {'North':<14} | {'South':<14} | {'East':<14} | {'West':<14} | {'Audio Siren':<16} | Status")
    print("-" * 90)

    start_time = time.perf_counter()
    step = 0
    try:
        while True:
            step += 1
            states = manager.step()

            # Process audio window periodically (approx 1s cadence)
            if audio_windows_gen and (step % 5 == 1):
                try:
                    win = next(audio_windows_gen)
                    pred = classifier.predict(win)
                    latest_audio_state = smoother.update(pred)
                except StopIteration:
                    audio_input.open()
                    audio_windows_gen = audio_input.windows()
                except Exception as err:
                    logger.warning("Audio read error: %s", err)

            decision = fusion.evaluate(latest_audio_state, states)

            elapsed = time.perf_counter() - start_time
            if max_seconds and elapsed >= max_seconds:
                break

            # Print snapshot every step
            def dir_str(d_id):
                rep = states[d_id].traffic_report
                if not rep:
                    return "offline"
                return f"{rep.visible_now}v ({rep.level[:3]})"

            siren_text = f"{latest_audio_state.state[:8]} ({latest_audio_state.smoothed_confidence:.2f})"
            status_text = decision.headline
            print(f"{elapsed:>5.1f}s | {dir_str('north'):<14} | {dir_str('south'):<14} | "
                  f"{dir_str('east'):<14} | {dir_str('west'):<14} | {siren_text:<16} | {status_text}")

            time.sleep(0.08)

    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        manager.close()
        if audio_input:
            audio_input.close()
        logger.info("Pipeline terminated cleanly.")


def main():
    parser = argparse.ArgumentParser(description="SmartCity Sentinel Main Entrypoint")
    parser.add_argument("--dashboard", action="store_true", help="Launch interactive Streamlit dashboard")
    parser.add_argument("--audio", help="Override audio source (file or 'mic')")
    parser.add_argument("--seconds", type=float, help="Run for this many seconds then exit")
    args = parser.parse_args()

    config = load_config()
    ensure_directories(config)
    os.environ.setdefault("HF_HOME", str(config["paths"]["hf_cache_dir"]))
    setup_logging(config)

    if args.dashboard:
        from app import main as run_dashboard
        run_dashboard()
    else:
        run_pipeline(config, audio_source=args.audio, max_seconds=args.seconds)


if __name__ == "__main__":
    main()
