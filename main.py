"""
SmartCity Sentinel - main entry point.

Module 0: loads configuration, sets up logging and prepares folders.
The full video + audio pipeline is connected here in Module 12.

Run:
    python main.py
"""

import os

from app.utils.config import ensure_directories, load_config
from app.utils.logger import get_logger, setup_logging


def main():
    config = load_config()
    ensure_directories(config)

    # Hugging Face models download into the project, not the C: drive cache
    os.environ.setdefault("HF_HOME", str(config["paths"]["hf_cache_dir"]))

    setup_logging(config)
    logger = get_logger("main")

    logger.info("%s v%s starting", config["project"]["name"], config["project"]["version"])
    logger.info("Project folders ready (data, models, logs)")
    logger.info("Device preference from config: %s", config["device"])
    logger.info("Pipeline modules not yet implemented - foundation only (Module 0)")


if __name__ == "__main__":
    main()
