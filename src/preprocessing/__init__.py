"""Medical image preprocessing for fundus photographs.

Typical usage inside a notebook::

    from src.preprocessing import preprocess_image
    from src.utils.config import load_yaml

    data_cfg = load_yaml("data_config.yaml")
    clean = preprocess_image(raw_rgb, config=data_cfg)
"""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from src.preprocessing.circular_mask import apply_circular_mask, build_circular_mask
from src.preprocessing.enhancement import apply_clahe
from src.preprocessing.roi_cropper import crop_black_border

__all__ = [
    "apply_clahe",
    "apply_circular_mask",
    "build_circular_mask",
    "crop_black_border",
    "preprocess_image",
]


def preprocess_image(
    image: np.ndarray,
    image_size: int = 384,
    config: Mapping[str, Any] | None = None,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Run the full preprocessing chain and return a ``(size, size, 3)`` image.

    Order of operations::

        crop black border -> CLAHE -> circular mask -> resize

    CLAHE runs before the mask so the black corners cannot bias the local
    histogram, and the mask is applied before the resize so the border stays
    perfectly black after interpolation.

    Parameters
    ----------
    image:
        RGB ``uint8`` array straight out of ``cv2.imread`` after
        ``cv2.cvtColor(..., cv2.COLOR_BGR2RGB)``.
    image_size:
        Side of the output square (384 by default).
    config:
        The loaded ``data_config.yaml``. When ``None`` the preprocessing steps
        run with their default parameters.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise TypeError("preprocess_image expects a numpy array.")
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}.")

    steps: Mapping[str, Any] = {}
    if config:
        steps = config.get("preprocessing", {}) or {}

    roi_cfg = dict(steps.get("roi", {}) or {})
    clahe_cfg = dict(steps.get("clahe", {}) or {})
    mask_cfg = dict(steps.get("circular_mask", {}) or {})

    result = image
    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    elif result.shape[2] == 4:
        result = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)

    if roi_cfg.get("enabled", True):
        result = crop_black_border(
            result,
            threshold=int(roi_cfg.get("threshold", 7)),
            margin=int(roi_cfg.get("margin", 0)),
            min_area_ratio=float(roi_cfg.get("min_area_ratio", 0.05)),
        )

    if clahe_cfg.get("enabled", True):
        result = apply_clahe(
            result,
            clip_limit=float(clahe_cfg.get("clip_limit", 2.0)),
            tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", (8, 8))),
        )

    if mask_cfg.get("enabled", True):
        result = apply_circular_mask(
            result,
            radius_ratio=float(mask_cfg.get("radius_ratio", 0.5)),
        )

    return cv2.resize(result, (int(image_size), int(image_size)), interpolation=interpolation)
