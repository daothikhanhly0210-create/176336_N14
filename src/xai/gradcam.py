"""Grad-CAM explanations for the three backbones.

``pytorch-grad-cam`` is imported lazily so the rest of the project keeps working
on a machine where the package is not installed. ``xai_config.yaml`` stores the
target layer per architecture (``layer4[-1]``, ``features.denseblock4``,
``features[-1]``) and :func:`build_explainer` resolves it automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.models.architectures import DEFAULT_TARGET_LAYERS, resolve_target_layer

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_COLORMAPS = {
    "jet": "COLORMAP_JET",
    "turbo": "COLORMAP_TURBO",
    "inferno": "COLORMAP_INFERNO",
    "magma": "COLORMAP_MAGMA",
    "viridis": "COLORMAP_VIRIDIS",
}

__all__ = [
    "GradCAMExplainer",
    "build_explainer",
    "overlay_cam",
    "denormalize",
    "sanity_check",
]


def _colormap_code(name: str) -> int:
    if cv2 is None:
        return 0
    attribute = _COLORMAPS.get(str(name).lower(), "COLORMAP_JET")
    return getattr(cv2, attribute, cv2.COLORMAP_JET)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised ``(3, H, W)`` tensor back to a ``uint8`` RGB array."""
    array = tensor.detach().cpu().float().numpy()
    if array.ndim == 4:
        array = array[0]
    array = np.transpose(array, (1, 2, 0)) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def overlay_cam(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a ``[0, 1]`` heatmap over an RGB image."""
    if image_rgb is None or cam is None:
        raise ValueError("overlay_cam expects both an image and a CAM.")
    if cv2 is None:  # pragma: no cover - PIL fallback
        heat = (np.clip(cam, 0, 1)[..., None] * np.array([[255, 0, 0]])).astype(np.uint8)
        blended = 0.6 * image_rgb + 0.4 * heat
        return np.clip(blended, 0, 255).astype(np.uint8)

    height, width = image_rgb.shape[:2]
    heatmap = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), _colormap_code(colormap))
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)

    blended = (1.0 - float(alpha)) * image_rgb.astype(np.float32) + float(alpha) * heatmap
    return np.clip(blended, 0, 255).astype(np.uint8)


class GradCAMExplainer:
    """Thin wrapper around ``pytorch_grad_cam`` bound to one target layer."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | str,
        method: str = "gradcam",
        device: str | torch.device | None = None,
    ) -> None:
        try:
            import pytorch_grad_cam  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pytorch-grad-cam is required for XAI. Install it with "
                "`pip install grad-cam`."
            ) from exc

        self.model = model
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device).eval()
        self.target_layer = resolve_target_layer(self.model, target_layer)
        self.method = str(method).lower()

        self.cam = self._build_cam(self.method)

    def _build_cam(self, method: str):
        from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus, ScoreCAM

        registry = {
            "gradcam": GradCAM,
            "gradcam++": GradCAMPlusPlus,
            "gradcampp": GradCAMPlusPlus,
            "scorecam": ScoreCAM,
            "eigencam": EigenCAM,
        }
        if method not in registry:
            raise ValueError(f"Unknown CAM method '{method}'. Choose from {list(registry)}.")
        return registry[method](model=self.model, target_layers=[self.target_layer])

    def explain(self, image_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        """Return the ``(H, W)`` CAM for one batch of size 1."""
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.size(0) != 1:
            raise ValueError("explain() handles one image at a time; loop over the batch.")

        tensor = image_tensor.to(self.device)
        targets = None if target_class is None else [ClassifierOutputTarget(int(target_class))]

        with self.cam:
            grayscale = self.cam(input_tensor=tensor, targets=targets)

        grayscale = np.asarray(grayscale, dtype=np.float32)
        if grayscale.ndim == 3:
            grayscale = grayscale[0]
        return np.clip(grayscale, 0.0, 1.0)

    def explain_pil(
        self,
        image: Image.Image | np.ndarray,
        target_class: int | None = None,
        image_size: int = 384,
    ) -> np.ndarray:
        """Explain an image given as PIL/ndarray (applies ImageNet normalisation)."""
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"))
        else:
            array = np.asarray(image)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)

        if cv2 is not None:
            array = cv2.resize(array, (image_size, image_size))
        else:  # pragma: no cover
            array = np.asarray(Image.fromarray(array).resize((image_size, image_size)))

        normalised = (array.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(normalised).permute(2, 0, 1).unsqueeze(0).float()
        return self.explain(tensor, target_class=target_class)


def build_explainer(
    model: nn.Module,
    model_name: str | None = None,
    xai_config: Mapping[str, Any] | None = None,
    target_layer: str | nn.Module | None = None,
    device: str | torch.device | None = None,
) -> GradCAMExplainer:
    """Create an explainer, resolving the target layer from ``xai_config.yaml``."""
    gradcam_cfg = dict((xai_config or {}).get("gradcam", {}) or {})
    method = str(gradcam_cfg.get("method", "gradcam"))

    spec = target_layer
    if spec is None:
        mapping = dict(gradcam_cfg.get("target_layers", {}) or {})
        if model_name is not None and model_name in mapping:
            spec = mapping[model_name]
        elif model_name is not None and model_name in DEFAULT_TARGET_LAYERS:
            spec = DEFAULT_TARGET_LAYERS[model_name]
        elif len(mapping) == 1:
            spec = next(iter(mapping.values()))
        else:
            raise ValueError(
                "Cannot infer the Grad-CAM target layer. Pass `target_layer=` or "
                f"set gradcam.target_layers.{model_name} in xai_config.yaml."
            )

    return GradCAMExplainer(model, spec, method=method, device=device)


def sanity_check(
    model: nn.Module,
    loader: Any,
    target_layer: nn.Module | str,
    out_path: str | Path = "results/figures/gradcam_sanity_check.png",
    num_samples: int = 4,
    class_names: Sequence[str] | None = None,
    device: str | torch.device | None = None,
    alpha: float = 0.4,
    method: str = "gradcam",
    dpi: int = 200,
):
    """Save a grid ``input | heatmap | overlay`` and return the sampled rows.

    This is the figure used in the report to argue that the network looks at the
    lesion areas (haemorrhages, exudates) and not at the black border.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    explainer = GradCAMExplainer(model, target_layer, method=method, device=device)
    names = list(class_names or [])

    rows: List[Dict[str, Any]] = []
    for batch in loader:
        images = batch["image"] if isinstance(batch, Mapping) else batch[0]
        labels = batch["label"] if isinstance(batch, Mapping) else batch[1]
        paths = batch.get("image_path") if isinstance(batch, Mapping) else None

        for index in range(images.size(0)):
            if len(rows) >= num_samples:
                break
            tensor = images[index : index + 1]
            label = int(labels[index])
            prediction = int(model(tensor.to(explainer.device)).argmax(dim=1).item())
            cam = explainer.explain(tensor, target_class=prediction)
            rgb = denormalize(tensor[0])
            rows.append(
                {
                    "image": rgb,
                    "cam": cam,
                    "overlay": overlay_cam(rgb, cam, alpha=alpha),
                    "label": label,
                    "prediction": prediction,
                    "image_path": paths[index] if paths is not None else None,
                }
            )
        if len(rows) >= num_samples:
            break

    if not rows:
        raise RuntimeError("The loader is empty: cannot build the Grad-CAM sanity check.")

    figure, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    axes = np.atleast_2d(axes)
    for row_index, row in enumerate(rows):
        axes[row_index, 0].imshow(row["image"])
        axes[row_index, 0].set_title("Ảnh đầu vào", fontsize=9)
        axes[row_index, 1].imshow(row["cam"], cmap="jet")
        axes[row_index, 1].set_title("Heatmap Grad-CAM", fontsize=9)
        axes[row_index, 2].imshow(row["overlay"])
        true_name = names[row["label"]] if row["label"] < len(names) else row["label"]
        pred_name = (
            names[row["prediction"]] if row["prediction"] < len(names) else row["prediction"]
        )
        axes[row_index, 2].set_title(f"Overlay\ntrue: {true_name} | pred: {pred_name}", fontsize=9)
        for column in range(3):
            axes[row_index, column].axis("off")

    figure.tight_layout()
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return rows
