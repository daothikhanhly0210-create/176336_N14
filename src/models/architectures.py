"""Backbones used in the comparison: ResNet18, DenseNet121, EfficientNet-B0.

All three are initialised with ImageNet weights and the classifier is replaced by
``Dropout -> Linear(num_classes)``. The sequential head keeps the ``state_dict``
keys stable (``fc.1.weight``, ``classifier.1.weight``) whatever the dropout value,
which makes checkpoints portable.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from torchvision import models

MODEL_NAMES = ("resnet18", "densenet121", "efficientnet_b0")

#: Final convolutional stage of each backbone -> used by Grad-CAM.
DEFAULT_TARGET_LAYERS: Dict[str, str] = {
    "resnet18": "layer4[-1]",
    "densenet121": "features.denseblock4",
    "efficientnet_b0": "features[-1]",
}

#: Attribute that holds the classification head.
_HEAD_ATTRS: Dict[str, str] = {
    "resnet18": "fc",
    "densenet121": "classifier",
    "efficientnet_b0": "classifier",
}

__all__ = [
    "MODEL_NAMES",
    "DEFAULT_TARGET_LAYERS",
    "build_model",
    "resolve_target_layer",
    "freeze_backbone",
    "unfreeze_model",
    "count_parameters",
    "load_model_weights",
]


def _prepare_head(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Dropout(p=float(dropout)), nn.Linear(in_features, int(num_classes)))


def build_model(
    name: str,
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """Instantiate a backbone with a fresh classification head.

    Parameters
    ----------
    name:
        One of ``resnet18``, ``densenet121``, ``efficientnet_b0``.
    num_classes:
        Number of DR grades (5).
    pretrained:
        Load ImageNet weights (downloaded once and cached by torchvision).
    dropout:
        Dropout probability before the final linear layer.
    """
    key = str(name).lower().strip()
    if key not in MODEL_NAMES:
        raise ValueError(f"Unsupported model '{name}'. Choose from {MODEL_NAMES}.")

    weights = "DEFAULT" if pretrained else None
    try:
        model = models.get_model(key, weights=weights)
    except Exception as exc:  # pragma: no cover - depends on the local cache
        hint = (
            "Could not build the model. If the machine is offline, call "
            "build_model(..., pretrained=False) or pre-download the weights."
        )
        raise RuntimeError(f"{hint} Original error: {exc}") from exc

    if key == "resnet18":
        in_features = model.fc.in_features
        model.fc = _prepare_head(in_features, num_classes, dropout)
    elif key == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = _prepare_head(in_features, num_classes, dropout)
    else:  # efficientnet_b0
        in_features = model.classifier[1].in_features
        model.classifier = _prepare_head(in_features, num_classes, dropout)

    return model


def resolve_target_layer(model: nn.Module, spec: str | nn.Module | Any) -> nn.Module:
    """Resolve ``"layer4[-1]"`` / ``"features.denseblock4"`` into a real module.

    ``spec`` may also be an ``nn.Module`` already, in which case it is returned
    unchanged. This keeps ``xai_config.yaml`` free of Python code.
    """
    if isinstance(spec, nn.Module):
        return spec

    target: Any = model
    for part in str(spec).split("."):
        index = None
        if part.endswith("]") and "[" in part:
            part, raw_index = part[:-1].split("[", 1)
            index = int(raw_index)
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            raise AttributeError(
                f"'{part}' not found while resolving target layer '{spec}'. "
                f"Check xai_config.yaml for {type(model).__name__}."
            ) from exc
        if index is not None:
            target = target[index]

    if not isinstance(target, nn.Module):
        raise TypeError(f"Target layer '{spec}' resolved to {type(target)!r}, expected nn.Module.")
    return target


def freeze_backbone(model: nn.Module, name: str, freeze: bool = True) -> nn.Module:
    """Freeze every parameter except the classification head (linear probing)."""
    key = str(name).lower()
    if key not in _HEAD_ATTRS:
        raise ValueError(f"Unsupported model '{name}'. Choose from {MODEL_NAMES}.")

    for parameter in model.parameters():
        parameter.requires_grad = not freeze

    head = getattr(model, _HEAD_ATTRS[key])
    for parameter in head.parameters():
        parameter.requires_grad = True
    return model


def unfreeze_model(model: nn.Module) -> nn.Module:
    """Re-enable gradients everywhere (used when fine-tuning after a warm-up)."""
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of (trainable) parameters, handy for the report table."""
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )


def load_model_weights(
    model: nn.Module,
    checkpoint_path: str,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Load weights saved by ``Trainer`` (raw ``state_dict`` or its checkpoint dict)."""
    from src.models.trainer import load_checkpoint

    load_checkpoint(checkpoint_path, model=model, device=device, strict=strict)
    return model
