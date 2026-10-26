"""Apply a circular mask so the four black corners never reach the network.

After ROI cropping the retina is still a disc inside a rectangle. The corners are
pure black and carry no information, but a CNN can still pick up their gradient
(an artefact correlated with the camera that produced the image). Masking them
keeps attention on the retina.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["apply_circular_mask", "build_circular_mask"]


def build_circular_mask(
    shape: tuple[int, int],
    radius_ratio: float = 0.5,
    center: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return a ``uint8`` mask (255 inside the disc, 0 outside)."""
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid mask shape {shape}.")
    if not 0.0 < float(radius_ratio) <= 0.5:
        raise ValueError(f"radius_ratio must be in (0, 0.5], got {radius_ratio}.")

    radius = max(1, int(round(min(height, width) * float(radius_ratio))))
    cx, cy = center if center is not None else (width // 2, height // 2)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), radius, 255, thickness=-1)
    return mask


def apply_circular_mask(
    image: np.ndarray,
    radius_ratio: float = 0.5,
    fill_value: int = 0,
    center: tuple[int, int] | None = None,
) -> np.ndarray:
    """Zero everything outside the inscribed circle of the image.

    Parameters
    ----------
    image:
        RGB ``(H, W, 3)`` or grayscale ``(H, W)`` image.
    radius_ratio:
        Radius as a fraction of the shortest side; ``0.5`` gives the inscribed
        circle, smaller values add a black ring.
    fill_value:
        Value written outside the mask (``0`` = black).
    center:
        Optional ``(x, y)`` centre in pixels; defaults to the image centre.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise TypeError("apply_circular_mask expects a numpy array.")
    if image.size == 0:
        raise ValueError("apply_circular_mask received an empty array.")
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape {image.shape}.")

    mask = build_circular_mask(image.shape[:2], radius_ratio=radius_ratio, center=center)
    outside = mask == 0

    output = image.copy()
    if image.ndim == 2:
        output[outside] = fill_value
    else:
        output[outside, :] = fill_value
    return output
