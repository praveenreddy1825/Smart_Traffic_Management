"""
Evaluate the siren classifier on the ESC-50 subset (real recordings).

    python scripts/download_esc50_subset.py          # once
    python scripts/evaluate_siren_classifier.py      # uses window/hop from config
    python scripts/evaluate_siren_classifier.py --window-seconds 4 --device cpu

Method (avoids tuning on the test data):
  * every clip is cut into windows exactly like the live system (Module 6)
  * windows of near-digital silence are skipped (ESC-50 pads some clips with
    silence, so such windows contain no siren even in a "siren" clip)
  * the threshold is CHOSEN on folds 1-2 (best balanced accuracy)
  * results are REPORTED on folds 3-5 only
Results are saved to data/test/output/siren_eval_<window>s.json.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.audio.audio_input import FileAudioInput  # noqa: E402
from app.audio.siren_classifier import SirenClassifier  # noqa: E402
from app.utils.config import load_config  # noqa: E402
from app.utils.logger import setup_logging  # noqa: E402

CALIBRATION_FOLDS = {"1", "2"}
TEST_FOLDS = {"3", "4", "5"}


def metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(scores) >= threshold
    tp = int(np.sum(predicted & labels))
    tn = int(np.sum(~predicted & ~labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    recall = tp / max(tp + fn, 1)            # share of real sirens detected
    specificity = tn / max(tn + fp, 1)       # share of non-sirens correctly ignored
    return {
        "threshold": round(float(threshold), 3), "windows": int(len(labels)),
        "accuracy": (tp + tn) / max(len(labels), 1),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": tp / max(tp + fp, 1), "recall": recall,
        "false_positive_rate": fp / max(fp + tn, 1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def choose_threshold(labels, scores):
    """Best balanced accuracy on the calibration folds; ties -> higher threshold (fewer false alarms)."""
    best = None
    for threshold in np.round(np.arange(0.01, 0.96, 0.01), 2):
        result = metrics(labels, scores, threshold)
        if best is None or result["balanced_accuracy"] >= best["balanced_accuracy"]:
            best = result
    return best["threshold"]


def print_metrics(title, result):
    print(f"\n{title}")
    print(f"  threshold {result['threshold']:.2f} | windows {result['windows']} | "
          f"accuracy {result['accuracy']:.3f} | balanced acc {result['balanced_accuracy']:.3f}")
    print(f"  precision {result['precision']:.3f} | recall (sirens found) {result['recall']:.3f} | "
          f"false-positive rate {result['false_positive_rate']:.3f}")
    print(f"  TP {result['tp']}  FN {result['fn']}  |  TN {result['tn']}  FP {result['fp']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate siren detection on ESC-50 (Module 7)")
    parser.add_argument("--window-seconds", type=float, help="override audio.window_seconds")
    parser.add_argument("--hop-seconds", type=float, help="override audio.hop_seconds")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-clips", type=int, help="only use the first N clips (quick test)")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)
    if args.device:
        config["device"] = args.device
    window_seconds = args.window_seconds or config["audio"]["window_seconds"]
    hop_seconds = args.hop_seconds or min(config["audio"]["hop_seconds"], window_seconds)
    silence_dbfs = config["audio"].get("silence_threshold_dbfs", -50.0)

    folder = Path(config["paths"]["audio_dir"]) / "esc50"
    labels_file = folder / "labels.csv"
    if not labels_file.exists():
        print("Run scripts/download_esc50_subset.py first.")
        return 1
    clips = list(csv.DictReader(open(labels_file, encoding="utf-8")))[:args.max_clips]

    classifier = SirenClassifier.from_config(config)
    records, latencies, clip_records, skipped = [], [], [], 0
    for number, clip in enumerate(clips, 1):
        source = FileAudioInput(folder / clip["filename"], realtime_playback=False,
                                sample_rate=classifier.sample_rate,
                                window_seconds=window_seconds, hop_seconds=hop_seconds)
        source.open()
        for window in source.windows():
            if window.is_quiet(silence_dbfs):
                skipped += 1
                continue
            prediction = classifier.predict(window)
            latencies.append(prediction.inference_ms)
            records.append((clip["fold"], clip["is_siren"] == "1", prediction.siren_score,
                            clip["category"]))
        # Clip-level reference: the whole 5 s clip as ONE input (model's best case)
        whole = classifier.predict(samples=source.audio, sample_rate=source.sample_rate)
        clip_records.append((clip["fold"], clip["is_siren"] == "1", whole.siren_score))
        print(f"\rclassified {number}/{len(clips)} clips", end="", flush=True)
    print(f"\nWindows: {len(records)} analysed, {skipped} skipped as silent "
          f"(< {silence_dbfs} dBFS)  |  window {window_seconds}s, hop {hop_seconds}s")

    def subset(rows, folds):
        chosen = [row for row in rows if row[0] in folds]
        return [row[1] for row in chosen], [row[2] for row in chosen]

    calibration_labels, calibration_scores = subset(records, CALIBRATION_FOLDS)
    test_labels, test_scores = subset(records, TEST_FOLDS)
    chosen = choose_threshold(calibration_labels, calibration_scores)
    configured = config["siren"]["detection_threshold"]

    print_metrics(f"Threshold chosen on folds 1-2: {chosen:.2f}  (configured: {configured})",
                  metrics(calibration_labels, calibration_scores, chosen))
    test_chosen = metrics(test_labels, test_scores, chosen)
    test_configured = metrics(test_labels, test_scores, configured)
    print_metrics("TEST folds 3-5, windows, CHOSEN threshold", test_chosen)
    print_metrics("TEST folds 3-5, windows, CONFIGURED threshold", test_configured)
    clip_labels, clip_scores = subset(clip_records, TEST_FOLDS)
    test_clips = metrics(clip_labels, clip_scores, chosen)
    print_metrics("TEST folds 3-5, whole 5 s clips (reference), CHOSEN threshold", test_clips)

    test_rows = [row for row in records if row[0] in TEST_FOLDS]
    false_alarms = Counter(row[3] for row in test_rows if not row[1] and row[2] >= chosen)
    siren_scores = [row[2] for row in test_rows if row[1]]
    other_scores = [row[2] for row in test_rows if not row[1]]
    print(f"\nMean siren score (test): siren windows {np.mean(siren_scores):.3f} | "
          f"other windows {np.mean(other_scores):.3f}")
    print(f"False alarms by class (test, chosen threshold): {dict(false_alarms) or 'none'}")
    print(f"Inference latency per window on {classifier.device}: mean "
          f"{np.mean(latencies):.1f} ms, p95 {np.percentile(latencies, 95):.1f} ms "
          f"({len(latencies)} windows)")

    output = Path(config["paths"]["test_data_dir"]) / "output"
    output.mkdir(parents=True, exist_ok=True)
    result_file = output / f"siren_eval_{window_seconds:g}s.json"
    result_file.write_text(json.dumps({
        "model": classifier.model_name, "device": classifier.device,
        "window_seconds": window_seconds, "hop_seconds": hop_seconds,
        "windows_analysed": len(records), "windows_skipped_silent": skipped,
        "chosen_threshold": chosen, "test_windows_chosen": test_chosen,
        "test_windows_configured": test_configured, "test_whole_clips_chosen": test_clips,
        "false_alarms_by_class": dict(false_alarms),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
    }, indent=2))
    print(f"Saved {result_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
