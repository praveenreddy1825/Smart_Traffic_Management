"""
Launcher script for the SmartCity Sentinel High-Performance Web Dashboard.

Usage:
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 8000
    python scripts/run_dashboard.py --host 127.0.0.1 --port 8080
"""

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

# Ensure the script runs using the project's virtual environment (.venv)
ROOT_DIR = Path(__file__).resolve().parents[1]
venv_python = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
if venv_python.is_file() and sys.executable.lower() != str(venv_python).lower():
    sys.exit(subprocess.call([str(venv_python)] + sys.argv))

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn
from app.dashboard.server import app, pipeline


def sigint_handler(sig, frame):
    print("\n[INFO] Shutdown signal received, exiting cleanly...")
    try:
        pipeline.stop()
    except Exception:
        pass
    os._exit(0)


def main():
    try:
        signal.signal(signal.SIGINT, sigint_handler)
        signal.signal(signal.SIGTERM, sigint_handler)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Run SmartCity Sentinel Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on (default: 8000)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" 🚦 SmartCity Sentinel - High-Performance Web Dashboard")
    print(f" Dashboard URL: http://127.0.0.1:{args.port}")
    print("=" * 70 + "\n")

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_graceful_shutdown=0.1,
        timeout_keep_alive=3,
    )
    server = uvicorn.Server(config)
    try:
        server.run()
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
