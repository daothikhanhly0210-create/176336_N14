"""Console + file logger shared by every script and notebook."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_FILE = PROJECT_ROOT / "results" / "train.log"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

__all__ = ["get_logger", "DEFAULT_LOG_FILE"]


def get_logger(
    name: str = "brset",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a configured logger (idempotent).

    Parameters
    ----------
    name:
        Logger name; use one logger per module (``get_logger(__name__)``).
    log_file:
        Destination file. Defaults to ``results/train.log``; pass ``False`` to
        disable file logging (useful inside unit tests).
    level:
        Logging level for the logger itself.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if getattr(logger, "_brset_configured", False):
        return logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not False:
        target = Path(log_file) if log_file else DEFAULT_LOG_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger._brset_configured = True  # type: ignore[attr-defined]
    return logger
