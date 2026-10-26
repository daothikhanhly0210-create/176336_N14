"""Unit tests for the preprocessing pipeline (ROI, CLAHE, mask, resize)."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import (
    apply_clahe,
    apply_circular_mask,
    crop_black_border,
    preprocess_image,
)


@pytest.fixture()
def fundus_image() -> np.ndarray:
    """Synthetic fundus: a reddish disc on a black 800x600 frame."""
    height, width = 600, 800
    image = np.zeros((height, width, 3), dtype=np.uint8)
    rows, cols = np.mgrid[0:height, 0:width]
    distance = (rows - 300) ** 2 + (cols - 400) ** 2
    disc = distance <= 250**2
    gradient = (cols.astype(np.float32) / width * 60).astype(np.uint8)
    image[disc, 0] = 150
    image[disc, 1] = np.clip(70 + gradient[disc], 0, 255)
    image[disc, 2] = 40
    return image


def test_crop_black_border_removes_the_frame(fundus_image: np.ndarray) -> None:
    cropped = crop_black_border(fundus_image, threshold=7)

    assert cropped.ndim == 3 and cropped.shape[2] == 3
    assert cropped.shape[0] < fundus_image.shape[0]
    assert cropped.shape[1] < fundus_image.shape[1]
    # the disc touches y=50..550 and x=150..650 -> the crop must be ~500x500
    assert 480 <= cropped.shape[0] <= 520
    assert 480 <= cropped.shape[1] <= 520


def test_crop_black_border_keeps_all_black_image() -> None:
    black = np.zeros((120, 160, 3), dtype=np.uint8)
    cropped = crop_black_border(black)
    assert cropped.shape == black.shape


def test_crop_black_border_rejects_empty_array() -> None:
    with pytest.raises(ValueError):
        crop_black_border(np.zeros((0, 0, 3), dtype=np.uint8))


def test_apply_clahe_preserves_shape_and_dtype(fundus_image: np.ndarray) -> None:
    enhanced = apply_clahe(fundus_image, clip_limit=2.0, tile_grid_size=(8, 8))
    assert enhanced.shape == fundus_image.shape
    assert enhanced.dtype == np.uint8
    # CLAHE must actually change the contrast of the retina
    assert enhanced[300, 400].tolist() != fundus_image[300, 400].tolist()


def test_apply_clahe_rejects_bad_clip_limit(fundus_image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        apply_clahe(fundus_image, clip_limit=0)


def test_apply_clahe_supports_grayscale() -> None:
    gray = np.full((64, 64), 120, dtype=np.uint8)
    assert apply_clahe(gray).shape == gray.shape


def test_circular_mask_blacks_out_the_corners(fundus_image: np.ndarray) -> None:
    masked = apply_circular_mask(fundus_image, radius_ratio=0.5)
    assert masked.shape == fundus_image.shape
    for corner in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        assert masked[corner].tolist() == [0, 0, 0]
    assert masked[300, 400].tolist() != [0, 0, 0]


def test_circular_mask_validates_radius(fundus_image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        apply_circular_mask(fundus_image, radius_ratio=0.0)
    with pytest.raises(ValueError):
        apply_circular_mask(fundus_image, radius_ratio=0.8)


def test_preprocess_image_outputs_384_square(fundus_image: np.ndarray) -> None:
    config = {
        "preprocessing": {
            "roi": {"enabled": True, "threshold": 7, "margin": 0, "min_area_ratio": 0.05},
            "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
            "circular_mask": {"enabled": True, "radius_ratio": 0.5},
        }
    }
    processed = preprocess_image(fundus_image, image_size=384, config=config)

    assert processed.shape == (384, 384, 3)
    assert processed.dtype == np.uint8
    assert processed[0, 0].tolist() == [0, 0, 0]


def test_preprocess_image_can_skip_steps(fundus_image: np.ndarray) -> None:
    config = {
        "preprocessing": {
            "roi": {"enabled": False},
            "clahe": {"enabled": False},
            "circular_mask": {"enabled": False},
        }
    }
    processed = preprocess_image(fundus_image, image_size=224, config=config)
    assert processed.shape == (224, 224, 3)


def test_preprocess_image_rejects_bad_size(fundus_image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        preprocess_image(fundus_image, image_size=0)
