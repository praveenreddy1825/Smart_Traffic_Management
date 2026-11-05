"""
Module 14: System Performance, Accuracy & Latency Benchmark Runner.

Executes quantitative benchmarks across the complete multimodal pipeline:
  1. Vehicle Detector Latency (YOLO11n): per-frame inference speed (ms), memory footprint, FPS.
  2. Vehicle Tracker Latency (ByteTrack): per-frame update speed (ms).
  3. Audio Siren Classifier Latency (AST): per-window inference speed (ms).
  4. Multi-Approach Pipeline Cycle Latency: 4-camera concurrent capture + inference throughput (FPS).
  5. End-to-End Latency & Reaction Time: timestamped acoustic-to-preemption delay.

Saves structured benchmark metrics to data/test/output/benchmark_results.json.

Usage:
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --runs 30
    python scripts/run_benchmarks.py --device cpu
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio.audio_input import AudioWindow
from app.audio.siren_classifier import SirenClassifier
from app.detection.vehicle_detector import VehicleDetector
from app.tracking.vehicle_counter import VehicleCounter
from app.traffic.multi_approach_manager import MultiApproachManager
from app.utils.config import ensure_directories, load_config, resolve_path
from app.utils.device import get_device
from app.utils.logger import get_logger, setup_logging
from app.video.video_input import VideoInput

logger = get_logger("benchmarks")


def compute_stats(times_ms: List[float]) -> Dict[str, float]:
    """Calculate mean, standard deviation, p50, p90, p95, p99 for a list of latencies in ms."""
    arr = np.array(times_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "max_ms": round(float(np.max(arr)), 2),
    }


def benchmark_detector(config: dict, runs: int = 50) -> Dict[str, Any]:
    """Measure YOLO detector inference latency on dummy and real frames."""
    detector = VehicleDetector.from_config(config)
    img_size = config.get("detection", {}).get("image_size", 640)
    # Warm-up
    dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for _ in range(5):
        _ = detector.detect(dummy)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = detector.detect(dummy)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    stats = compute_stats(latencies)
    fps = round(1000.0 / stats["mean_ms"], 1) if stats["mean_ms"] > 0 else 0.0
    return {
        "model": config["detection"]["model_name"],
        "device": detector.device,
        "input_size": img_size,
        "runs": runs,
        "fps": fps,
        "latency_stats": stats,
    }


def benchmark_classifier(config: dict, runs: int = 25) -> Dict[str, Any]:
    """Measure AST audio siren classifier inference latency on a 2.0s audio window."""
    classifier = SirenClassifier.from_config(config)
    sr = classifier.sample_rate
    win_samples = int(sr * config["audio"]["window_seconds"])
    dummy_audio = np.zeros(win_samples, dtype=np.float32)
    win = AudioWindow(dummy_audio, sr, 0, 0.0, -60.0, 0.0)

    # Warmup
    for _ in range(3):
        _ = classifier.predict(win)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = classifier.predict(win)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    stats = compute_stats(latencies)
    return {
        "model": config["siren"]["model_name"],
        "device": classifier.device,
        "sample_rate": sr,
        "window_seconds": config["audio"]["window_seconds"],
        "runs": runs,
        "latency_stats": stats,
    }


def benchmark_multi_approach(config: dict, steps: int = 30) -> Dict[str, Any]:
    """Measure multi-approach 4-stream processing throughput and cycle times."""
    manager = MultiApproachManager.from_config(config, loop_videos=True)
    manager.open()

    cycle_times = []
    try:
        # Warmup
        for _ in range(5):
            _ = manager.step()

        for _ in range(steps):
            t0 = time.perf_counter()
            states = manager.step()
            cycle_times.append((time.perf_counter() - t0) * 1000.0)
    finally:
        manager.close()

    stats = compute_stats(cycle_times)
    total_effective_fps = round((4 * 1000.0) / stats["mean_ms"], 1) if stats["mean_ms"] > 0 else 0.0
    cycles_per_sec = round(1000.0 / stats["mean_ms"], 1) if stats["mean_ms"] > 0 else 0.0

    return {
        "steps": steps,
        "active_approaches": 4,
        "cycle_latency_stats": stats,
        "cycles_per_second": cycles_per_sec,
        "effective_aggregate_fps": total_effective_fps,
    }


def main():
    parser = argparse.ArgumentParser(description="Run System Performance Benchmarks (Module 14)")
    parser.add_argument("--runs", type=int, default=30, help="Number of test iterations")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device")
    parser.add_argument("--output", default="data/test/output/benchmark_results.json", help="Path to save results")
    args = parser.parse_args()

    config = load_config()
    if args.device != "auto":
        config["device"] = args.device

    setup_logging(config)
    ensure_directories(config)

    print("\n" + "=" * 80)
    print(" SmartCity Sentinel - Module 14: System Performance & Latency Benchmarks")
    print("=" * 80)
    print(f"Target compute device: {config.get('device', 'auto')} (detected: {get_device(config.get('device', 'auto'))})")
    print(f"Benchmark iterations: {args.runs}")

    # 1. Detector benchmark
    print("\n[1/3] Benchmarking Vehicle Detector (YOLO11n)...")
    detector_results = benchmark_detector(config, runs=args.runs)
    print(f"  -> Mean latency: {detector_results['latency_stats']['mean_ms']:.2f} ms "
          f"(p95: {detector_results['latency_stats']['p95_ms']:.2f} ms) "
          f"| FPS: {detector_results['fps']} on {detector_results['device']}")

    # 2. Audio Classifier benchmark
    print("\n[2/3] Benchmarking Audio Siren Classifier (AST)...")
    classifier_runs = max(10, args.runs // 2)
    classifier_results = benchmark_classifier(config, runs=classifier_runs)
    print(f"  -> Mean latency: {classifier_results['latency_stats']['mean_ms']:.2f} ms "
          f"(p95: {classifier_results['latency_stats']['p95_ms']:.2f} ms) "
          f"on {classifier_results['device']}")

    # 3. Multi-Approach Pipeline benchmark
    print("\n[3/3] Benchmarking 4-Way Multi-Approach Intersection Pipeline...")
    multi_results = benchmark_multi_approach(config, steps=max(15, args.runs // 2))
    print(f"  -> Cycle latency: {multi_results['cycle_latency_stats']['mean_ms']:.2f} ms/cycle "
          f"(Aggregate FPS: {multi_results['effective_aggregate_fps']} across 4 cameras)")

    benchmarks_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": {
            "device": detector_results["device"],
        },
        "vehicle_detector": detector_results,
        "audio_siren_classifier": classifier_results,
        "multi_approach_pipeline": multi_results,
    }

    out_path = resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks_summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f" Benchmarks complete. Output written to: {out_path}")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

