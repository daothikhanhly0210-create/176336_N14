"""Class weighting utilities for the imbalanced DR grading task.

BRSET is heavily skewed towards ``No DR`` / ``Moderate``. Plain cross-entropy
then collapses to the majority grades, which destroys the macro F1 of the rare
``Severe`` / ``Proliferative`` classes. We use inverse-frequency weights with a
square root damping (the standard "effective number" trick) so the loss is
rebalanced without over-sampling the rare classes to death.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = ["compute_class_weight", "build_criterion", "focal_loss"]


def compute_class_weight(
    labels: Sequence[int] | np.ndarray,
    num_classes: int | None = None,
    mode: str = "sqrt",
    normalize: bool = True,
) -> torch.Tensor:
    """Return per-class weights for ``nn.CrossEntropyLoss(weight=...)``.

    Parameters
    ----------
    labels:
        Integer labels of the training split (one per image).
    num_classes:
        Number of classes; inferred from the labels when ``None``.
    mode:
        ``"inverse"`` = ``N / (K * n_c)``, ``"sqrt"`` = square root of that
        (softer), ``"none"`` = all ones.
    normalize:
        Rescale the weights so they average to 1.0, keeping the loss magnitude
        comparable to an unweighted run (and therefore the learning rate valid).
    """
    array = np.asarray(labels)
    if array.size == 0:
        raise ValueError("compute_class_weight received an empty label array.")
    if array.ndim != 1:
        array = array.reshape(-1)
    if not np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.int64)

    num_classes = int(num_classes or (int(array.max()) + 1))
    counts = np.bincount(array, minlength=num_classes).astype(np.float64)
    if counts.sum() == 0:
        raise ValueError("All labels are out of range for the requested num_classes.")

    mode = str(mode).lower()
    if mode in ("none", "uniform", "false"):
        weights = np.ones(num_classes, dtype=np.float64)
    else:
        safe_counts = np.where(counts == 0, 1.0, counts)
        weights = counts.sum() / (num_classes * safe_counts)
        if mode == "sqrt":
            weights = np.sqrt(weights)
        elif mode != "inverse":
            raise ValueError(f"Unknown class weighting mode '{mode}' (none|sqrt|inverse).")
        # classes absent from the split must not be penalised
        weights[counts == 0] = 0.0

    if normalize and weights.sum() > 0:
        weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Multi-class focal loss (Lin et al., 2017) with optional class weights."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    nll = torch.nn.functional.nll_loss(
        log_probs,
        targets,
        weight=weight,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    probs = torch.exp(log_probs)
    focal_factor = (1.0 - probs.gather(1, targets.unsqueeze(1)).squeeze(1)) ** float(gamma)
    return (focal_factor * nll).mean()


def build_criterion(
    config: Mapping[str, Any] | None = None,
    class_weights: torch.Tensor | None = None,
    device: str | torch.device | None = None,
) -> nn.Module:
    """Build the loss from the ``training.loss`` section of ``model_config.yaml``."""
    loss_cfg = dict((config or {}).get("loss", {}) or {})
    name = str(loss_cfg.get("name", "cross_entropy")).lower()
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))

    weight = class_weights
    if weight is not None and device is not None:
        weight = weight.to(device)

    if name in ("cross_entropy", "ce"):
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    if name == "focal":
        gamma = float(loss_cfg.get("gamma", 2.0))

        class _FocalLoss(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = weight
                self.gamma = gamma
                self.label_smoothing = label_smoothing

            def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                return focal_loss(
                    logits,
                    targets,
                    weight=self.weight,
                    gamma=self.gamma,
                    label_smoothing=self.label_smoothing,
                )

        return _FocalLoss()

    raise ValueError(f"Unknown loss '{name}' (cross_entropy|focal).")
