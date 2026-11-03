"""Reproducibility helpers used across the entire AgriVerse pipeline."""
import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy, and torch (if available) RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(preference: str = "auto"):
    """Resolve a torch device, falling back to CPU when CUDA is unavailable."""
    import torch

    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
