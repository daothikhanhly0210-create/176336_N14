"""Crop the black border that surrounds retinal fundus photographs.

The raw BRSET images contain a large black frame. Feeding it to the CNN wastes
resolution: after ``Resize(384, 384)`` the useful retina only covers a fraction
of the input. Cropping the frame first keeps the lesion texture at full detail.
"""

from __future__ import annotations

from typing import Union

import cv2
import numpy as np

__all__ = ["crop_black_border"]


def _as_uint8(image: np.ndarray) -> np.ndarray:
    """Normalise an image to ``uint8`` without changing its shape."""
    if image.dtype == np.uint8:
        return image
    array = image.astype(np.float32)
    if float(array.max()) <= 1.0:
        array *= 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    raise ValueError(f"Unsupported image shape {image.shape}; expected (H, W) or (H, W, 3/4).")


def crop_black_border(
    image: np.ndarray,
    threshold: int = 7,
    margin: int = 0,
    min_area_ratio: float = 0.05,
    kernel_size: int = 5,
) -> np.ndarray:
    """Return the smallest axis-aligned crop that contains the retina.

    Parameters
    ----------
    image:
        RGB (or grayscale) image, any numeric dtype; ``uint8`` avoids conversion.
    threshold:
        Pixel intensity (0-255) above which a pixel is considered "content".
        BRSET borders are almost pure black, so 5-10 works well.
    margin:
        Extra pixels kept around the detected bounding box.
    min_area_ratio:
        If the detected box covers less than this fraction of the image it is
        probably noise (or a fully black image) and the input is returned as is.
    kernel_size:
        Size of the morphological opening kernel used to drop isolated pixels.

    Notes
    -----
    The function never fails on "empty" images: a fully black picture is
    returned unchanged so the pipeline can continue.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise TypeError("crop_black_border expects a numpy array.")
    if image.size == 0:
        raise ValueError("crop_black_border received an empty array.")
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape {image.shape}.")

    raw = _as_uint8(image)
    gray = _to_gray(raw)

    mask = (gray > int(threshold)).astype(np.uint8)
    if kernel_size and kernel_size > 1:
        kernel = np.ones((int(kernel_size), int(kernel_size)), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return raw.copy()

    y1, y2 = int(rows[0]), int(rows[-1])
    x1, x2 = int(cols[0]), int(cols[-1])

    height, width = gray.shape[:2]
    box_area = (y2 - y1 + 1) * (x2 - x1 + 1)
    if box_area / float(height * width) < float(min_area_ratio):
        return raw.copy()

    if margin:
        margin = max(0, int(margin))
        y1 = max(0, y1 - margin)
        x1 = max(0, x1 - margin)
        y2 = min(height - 1, y2 + margin)
        x2 = min(width - 1, x2 + margin)

    cropped = raw[y1 : y2 + 1, x1 : x2 + 1]
    if cropped.size == 0:
        return raw.copy()
    return np.ascontiguousarray(cropped)


def crop_batch(images: list[np.ndarray], **kwargs: Union[int, float]) -> list[np.ndarray]:
    """Apply :func:`crop_black_border` to a list of images."""
    return [crop_black_border(image, **kwargs) for image in images]
