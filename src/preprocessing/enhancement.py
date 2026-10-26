"""Contrast enhancement with CLAHE applied on the LAB lightness channel.

Fundus photographs are often unevenly illuminated: the macula area is darker and
small lesions (microaneurysms, haemorrhages) get lost. CLAHE (Contrast Limited
Adaptive Histogram Equalisation) stretches the local contrast without blowing up
the noise, and doing it in LAB keeps the colours of the retina natural.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

__all__ = ["apply_clahe"]


def _as_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    array = image.astype(np.float32)
    if float(array.max()) <= 1.0:
        array *= 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Sequence[int] | Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance local contrast of a fundus image.

    Parameters
    ----------
    image:
        RGB image ``(H, W, 3)`` or grayscale ``(H, W)``.
    clip_limit:
        Contrast threshold for CLAHE; ``2.0`` is a good default for fundus
        images, higher values amplify noise.
    tile_grid_size:
        Number of tiles per axis used by the adaptive histogram equalisation.

    Returns
    -------
    numpy.ndarray
        Image with the same shape/dtype as the input (``uint8``).
    """
    if image is None or not isinstance(image, np.ndarray):
        raise TypeError("apply_clahe expects a numpy array.")
    if image.size == 0:
        raise ValueError("apply_clahe received an empty array.")
    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be > 0, got {clip_limit}.")
    if len(tile_grid_size) != 2 or min(int(v) for v in tile_grid_size) < 1:
        raise ValueError(f"tile_grid_size must be two positive ints, got {tile_grid_size}.")

    raw = _as_uint8(image)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size[0]), int(tile_grid_size[1])),
    )

    if raw.ndim == 2:
        return clahe.apply(raw)

    if raw.ndim != 3 or raw.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image shape {raw.shape}.")

    has_alpha = raw.shape[2] == 4
    rgb = raw[:, :, :3]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge([lightness, a_channel, b_channel]), cv2.COLOR_LAB2RGB)

    if has_alpha:
        enhanced = np.dstack([enhanced, raw[:, :, 3]])
    return enhanced
