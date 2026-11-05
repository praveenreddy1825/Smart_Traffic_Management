"""
SmartCity Sentinel - Web Application Entrypoint.

Starts the high-performance FastAPI server and serves the responsive 4-way
multimodal traffic and emergency preemption dashboard.

Run directly via:
    python app.py
or:
    python scripts/run_dashboard.py
    python main.py --dashboard
"""

import os
import subprocess
import sys
from pathlib import Path

# Ensure the script runs using the project's virtual environment (.venv)
ROOT_DIR = Path(__file__).resolve().parent
venv_python = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
if venv_python.is_file() and sys.executable.lower() != str(venv_python).lower():
    sys.exit(subprocess.call([str(venv_python)] + sys.argv))

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn
from app.dashboard.server import app


def main(host: str = "127.0.0.1", port: int = 8000):
    print("\n" + "=" * 70)
    print(" 🚦 SmartCity Sentinel - High-Performance Web Dashboard")
    print(f" Dashboard URL: http://127.0.0.1:{port}")
    print("=" * 70 + "\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
