"""
Compute-device selection (GPU or CPU) shared by the YOLO and audio models.
"""

from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_device(preference="auto"):
    """Return "cuda" or "cpu" based on the config preference.

    "auto" uses the NVIDIA GPU when PyTorch can see one, otherwise the CPU.
    Asking for "cuda" on a machine without a GPU falls back to CPU with a warning.
    """
    import torch  # imported here so config/logging work even before torch is installed

    preference = str(preference).lower()
    cuda_available = torch.cuda.is_available()

    if preference == "cpu":
        return "cpu"
    if preference == "cuda" and not cuda_available:
        logger.warning("CUDA requested but no GPU is available - falling back to CPU")
        return "cpu"
    return "cuda" if cuda_available else "cpu"
