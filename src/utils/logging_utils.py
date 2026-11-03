"""Lightweight logging setup shared by every AgriVerse script."""
import logging
import os
import sys
from datetime import datetime


def get_logger(name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (avoid duplicate handlers)

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{name}_{timestamp}.log")
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
