"""
Download a labelled subset of the ESC-50 dataset for evaluating siren detection.

ESC-50: K. J. Piczak, "ESC: Dataset for Environmental Sound Classification",
ACM Multimedia 2015.  https://github.com/karolpiczak/ESC-50
License: Creative Commons Attribution-NonCommercial (CC BY-NC 3.0).
Clips are 5 s recordings from freesound.org. Used here only for
non-commercial academic evaluation.

Downloads:
  - ALL 40 'siren' clips
  - 1 clip per fold (5 per class) from each of 16 non-siren classes chosen
    because they are traffic-related or easily confused with sirens (80 clips)
into data/audio/esc50/ together with labels.csv.

    python scripts/download_esc50_subset.py
"""

import csv
import io
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.config import load_config  # noqa: E402

BASE_URL = "https://github.com/karolpiczak/ESC-50/raw/master"
SIREN_CLASS = "siren"
NEGATIVE_CLASSES = [
    # traffic / street sounds
    "car_horn", "engine", "train", "airplane", "helicopter", "footsteps", "rain", "wind",
    # tonal or alarm-like sounds that could be confused with a siren
    "clock_alarm", "church_bells", "crying_baby", "rooster", "cat", "vacuum_cleaner",
    "chainsaw", "laughing",
]


def download(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main():
    config = load_config()
    folder = Path(config["paths"]["audio_dir"]) / "esc50"
    folder.mkdir(parents=True, exist_ok=True)

    print("Downloading ESC-50 metadata...")
    rows = list(csv.DictReader(io.StringIO(download(f"{BASE_URL}/meta/esc50.csv").decode())))

    selected, per_class = [], defaultdict(int)
    for row in sorted(rows, key=lambda r: (int(r["fold"]), r["filename"])):
        category = row["category"]
        if category == SIREN_CLASS:
            selected.append(row)
        elif category in NEGATIVE_CLASSES and per_class[(category, row["fold"])] < 1:
            # one clip per class from EACH of the 5 folds, so every fold has negatives
            selected.append(row)
            per_class[(category, row["fold"])] += 1

    for number, row in enumerate(selected, 1):
        target = folder / row["filename"]
        if not target.exists():
            target.write_bytes(download(f"{BASE_URL}/audio/{row['filename']}"))
        print(f"\r{number}/{len(selected)} {row['filename']}", end="", flush=True)

    with open(folder / "labels.csv", "w", newline="", encoding="utf-8") as label_file:
        writer = csv.writer(label_file)
        writer.writerow(["filename", "fold", "category", "is_siren"])
        for row in selected:
            writer.writerow([row["filename"], row["fold"], row["category"],
                             int(row["category"] == SIREN_CLASS)])

    # Remove clips left over from an earlier, different selection
    wanted = {row["filename"] for row in selected}
    for old_file in folder.glob("*.wav"):
        if old_file.name not in wanted:
            old_file.unlink()

    sirens = sum(row["category"] == SIREN_CLASS for row in selected)
    print(f"\nSaved {len(selected)} clips ({sirens} siren, {len(selected) - sirens} non-siren) "
          f"+ labels.csv to {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
