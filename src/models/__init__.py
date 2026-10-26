"""Model factory, losses and training loop."""

from src.models.architectures import (
    DEFAULT_TARGET_LAYERS,
    MODEL_NAMES,
    build_model,
    count_parameters,
    freeze_backbone,
    load_model_weights,
    resolve_target_layer,
    unfreeze_model,
)
from src.models.losses import build_criterion, compute_class_weight, focal_loss
from src.models.trainer import (
    EarlyStopping,
    Trainer,
    build_optimizer,
    build_scheduler,
    load_checkpoint,
    predict_loader,
)

__all__ = [
    "MODEL_NAMES",
    "DEFAULT_TARGET_LAYERS",
    "build_model",
    "resolve_target_layer",
    "freeze_backbone",
    "unfreeze_model",
    "count_parameters",
    "load_model_weights",
    "compute_class_weight",
    "build_criterion",
    "focal_loss",
    "Trainer",
    "EarlyStopping",
    "build_optimizer",
    "build_scheduler",
    "load_checkpoint",
    "predict_loader",
]
