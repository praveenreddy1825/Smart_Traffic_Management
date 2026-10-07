"""
Environment check for SmartCity Sentinel.

Verifies that the Python version, libraries, GPU, webcam and microphone are
usable on this machine. Run it after installation:

    python scripts/check_environment.py            # full check
    python scripts/check_environment.py --no-camera  # skip opening the webcam
"""

import argparse
import importlib
import sys
from pathlib import Path

# Allow "import app..." when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.config import load_config  # noqa: E402

# (import name, display name)
REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("yaml", "pyyaml"),
    ("cv2", "opencv-python"),
    ("torch", "torch"),
    ("ultralytics", "ultralytics"),
    ("transformers", "transformers"),
    ("sounddevice", "sounddevice"),
    ("soundfile", "soundfile"),
    ("scipy", "scipy"),
    ("streamlit", "streamlit"),
    ("pytest", "pytest"),
]


def check_python():
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 10
    print(f"{'[OK]  ' if ok else '[FAIL]'} Python {version.major}.{version.minor}.{version.micro}")
    return ok


def check_packages():
    all_ok = True
    for import_name, display_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown version")
            print(f"[OK]   {display_name:<14} {version}")
        except ImportError as error:
            print(f"[FAIL] {display_name:<14} not installed ({error})")
            all_ok = False
    return all_ok


def check_gpu():
    try:
        import torch
    except ImportError:
        print("[FAIL] GPU check skipped - torch not installed")
        return False
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"[OK]   CUDA GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB), "
              f"CUDA {torch.version.cuda}")
    else:
        # Not a failure: the project also runs on CPU, just slower
        print("[WARN] No CUDA GPU visible to PyTorch - models will run on CPU")
    return True


def check_camera(camera_index=0):
    import cv2
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    try:
        if not capture.isOpened():
            print(f"[WARN] Webcam {camera_index} could not be opened (recorded video will still work)")
            return True
        success, frame = capture.read()
        if success:
            height, width = frame.shape[:2]
            print(f"[OK]   Webcam {camera_index}: frame {width}x{height}")
        else:
            print(f"[WARN] Webcam {camera_index} opened but returned no frame")
        return True
    finally:
        capture.release()


def check_microphone():
    import sounddevice as sd
    try:
        default_input = sd.default.device[0]
        input_devices = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
    except Exception as error:  # PortAudio errors when no audio system is present
        print(f"[WARN] Could not query audio devices: {error}")
        return True
    if not input_devices:
        print("[WARN] No microphone found (recorded audio files will still work)")
        return True
    name = sd.query_devices(default_input)["name"] if default_input >= 0 else "none"
    print(f"[OK]   Microphone: {len(input_devices)} input device(s), default = {name}")
    return True


def check_config():
    try:
        config = load_config()
        print(f"[OK]   Config loaded: {config['project']['name']} v{config['project']['version']}")
        return True
    except Exception as error:
        print(f"[FAIL] Config could not be loaded: {error}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check the SmartCity Sentinel environment")
    parser.add_argument("--no-camera", action="store_true", help="skip the webcam test")
    args = parser.parse_args()

    print("=== SmartCity Sentinel - environment check ===")
    results = [check_python(), check_config(), check_packages(), check_gpu()]
    if not args.no_camera:
        results.append(check_camera())
    results.append(check_microphone())

    print("=" * 46)
    if all(results):
        print("Environment ready.")
        return 0
    print("Environment has problems - see [FAIL] lines above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
