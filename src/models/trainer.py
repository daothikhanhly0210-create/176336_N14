"""Training loop: weighted loss, AMP, cosine schedule, early stopping, checkpoints.

The class is deliberately framework-free (plain PyTorch) so the same code runs in
a notebook on Colab and in a script on a local GPU/CPU.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics
from src.utils.config import resolve_project_path
from src.utils.logger import get_logger

__all__ = [
    "EarlyStopping",
    "Trainer",
    "load_checkpoint",
    "predict_loader",
    "build_optimizer",
    "build_scheduler",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    try:  # torch >= 2.3
        return torch.amp.autocast("cuda", enabled=True)
    except AttributeError:  # pragma: no cover - older torch
        return torch.cuda.amp.autocast(enabled=True)


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_optimizer(
    model: nn.Module, config: Mapping[str, Any] | None = None
) -> torch.optim.Optimizer:
    """AdamW (default) with the learning rate from ``model_config.yaml``."""
    cfg = dict((config or {}).get("optimizer", {}) or {})
    name = str(cfg.get("name", "adamw")).lower()
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    momentum = float(cfg.get("momentum", 0.9))

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True
        )
    raise ValueError(f"Unsupported optimizer '{name}' (adamw|adam|sgd).")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any] | None = None,
    epochs: int | None = None,
):
    """Cosine / step / none scheduler, driven per epoch."""
    cfg = dict((config or {}).get("scheduler", {}) or {})
    name = str(cfg.get("name", "cosine")).lower()
    if name in ("none", "false", ""):
        return None

    total_epochs = int(epochs or (config or {}).get("epochs", 30))
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs), eta_min=float(cfg.get("eta_min", 0.0))
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 10)),
            gamma=float(cfg.get("gamma", 0.1)),
        )
    raise ValueError(f"Unsupported scheduler '{name}' (cosine|step|none).")


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> Tuple[Dict[str, Any], nn.Module | None]:
    """Load a checkpoint saved by :class:`Trainer`.

    Accepts both the rich dict written by the trainer and a bare ``state_dict``.
    """
    path = resolve_project_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - torch < 2.0
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format in {path}: {type(checkpoint)!r}")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if model is not None:
        model.load_state_dict(state_dict, strict=strict)
        model.to(device)
    return checkpoint, model


@dataclass
class EarlyStopping:
    """Stop training when the monitored metric stops improving."""

    patience: int = 5
    mode: str = "max"
    min_delta: float = 0.0
    best: float = field(init=False)
    counter: int = field(default=0, init=False)
    should_stop: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'.")
        self.best = float("-inf") if self.mode == "max" else float("inf")

    def step(self, value: float) -> bool:
        """Register a new value; return ``True`` when it is an improvement."""
        value = float(value)
        if not np.isfinite(value):
            self.counter += 1
            self.should_stop = self.counter >= int(self.patience)
            return False

        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
            return True

        self.counter += 1
        self.should_stop = self.counter >= int(self.patience)
        return False


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device | None = None,
    return_meta: bool = False,
):
    """Run inference with a plain model (no ``Trainer`` instance required).

    Returns ``(y_true, y_pred)`` or, with ``return_meta=True``,
    ``(y_true, y_pred, meta)`` where each meta item holds ``image_path`` and
    ``patient_id`` (only available when the dataset uses ``return_meta=True``).
    """
    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(target_device).eval()

    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []

    for batch in loader:
        if isinstance(batch, Mapping):
            images = batch["image"]
            labels = batch["label"]
        else:
            images, labels = batch[0], batch[1]

        logits = model(images.to(target_device, non_blocking=True))
        y_true.append(np.asarray(labels))
        y_pred.append(logits.argmax(dim=1).cpu().numpy())

        if return_meta and isinstance(batch, Mapping):
            paths = batch.get("image_path")
            patients = batch.get("patient_id")
            for index in range(len(y_true[-1])):
                meta.append(
                    {
                        "image_path": paths[index] if paths is not None else None,
                        "patient_id": patients[index] if patients is not None else None,
                    }
                )

    y_true_array = np.concatenate(y_true)
    y_pred_array = np.concatenate(y_pred)
    if return_meta:
        return y_true_array, y_pred_array, meta
    return y_true_array, y_pred_array


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Train one backbone and keep the best checkpoint on ``monitor``.

    Example
    -------
    >>> trainer = Trainer(model, train_loader, val_loader, epochs=30, patience=5,
    ...                   checkpoint_path="results/models/resnet18_best.pth")
    >>> history = trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        device: str | torch.device | None = None,
        epochs: int = 30,
        patience: int = 5,
        monitor: str = "val_f1_macro",
        mode: str = "max",
        min_delta: float = 0.0,
        checkpoint_path: str | Path | None = None,
        history_path: str | Path | None = None,
        amp: bool = True,
        grad_clip: float = 0.0,
        num_classes: int = 5,
        class_names: List[str] | None = None,
        logger: Any | None = None,
    ) -> None:
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-4
        )
        self.scheduler = scheduler
        self.epochs = int(epochs)
        self.num_classes = int(num_classes)
        self.class_names = class_names
        self.monitor = str(monitor)
        self.mode = str(mode)
        self.grad_clip = float(grad_clip)
        self.logger = logger or get_logger(__name__)

        self.amp = bool(amp and self.device.type == "cuda")
        self.scaler = _make_grad_scaler(self.amp)

        self.early_stopping = EarlyStopping(
            patience=int(patience), mode=self.mode, min_delta=float(min_delta)
        )

        self.checkpoint_path = (
            resolve_project_path(checkpoint_path) if checkpoint_path else None
        )
        self.history_path = resolve_project_path(history_path) if history_path else None

        self.history: List[Dict[str, Any]] = []
        self.best_metrics: Dict[str, Any] | None = None
        self.best_epoch: int = 0
        self.start_epoch = 0

    # ------------------------------------------------------------------ internals
    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    @staticmethod
    def _scalar_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
        return {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }

    def _run_epoch(self, loader: DataLoader, training: bool) -> Dict[str, Any]:
        self.model.train(training)
        total_loss = 0.0
        total_samples = 0
        y_true: List[np.ndarray] = []
        y_pred: List[np.ndarray] = []

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                images, labels = batch[0], batch[1]
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with _autocast(self.device, self.amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                if training:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                batch_size = images.size(0)
                total_loss += float(loss.detach().item()) * batch_size
                total_samples += batch_size
                y_true.append(labels.detach().cpu().numpy())
                y_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        if total_samples == 0:
            raise RuntimeError("Empty data loader: nothing to iterate over.")

        metrics = compute_metrics(
            np.concatenate(y_true),
            np.concatenate(y_pred),
            num_classes=self.num_classes,
            class_names=self.class_names,
        )
        metrics["loss"] = total_loss / total_samples
        return metrics

    def train_one_epoch(self) -> Dict[str, Any]:
        return self._run_epoch(self.train_loader, training=True)

    def validate(self) -> Dict[str, Any]:
        return self._run_epoch(self.val_loader, training=False)

    def _save_checkpoint(self, epoch: int, score: float, metrics: Mapping[str, Any]) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "score": float(score),
            "monitor": self.monitor,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "val_metrics": self._scalar_metrics(metrics),
        }
        torch.save(payload, self.checkpoint_path)

    def _save_history(self) -> None:
        if self.history_path is None or not self.history:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.history).to_csv(self.history_path, index=False)

    # ---------------------------------------------------------------------- API
    def fit(self) -> pd.DataFrame:
        """Run the full training loop and return the per-epoch history."""
        self.logger.info(
            "Training on %s | %d epochs | batch=%s | amp=%s | monitor=%s",
            self.device,
            self.epochs,
            getattr(self.train_loader, "batch_size", "?"),
            self.amp,
            self.monitor,
        )

        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            started = time.time()
            train_metrics = self.train_one_epoch()
            val_metrics = self.validate()
            if self.scheduler is not None:
                self.scheduler.step()

            row: Dict[str, Any] = {"epoch": epoch, "lr": self._current_lr(), "time_s": 0.0}
            row.update({f"train_{k}": v for k, v in self._scalar_metrics(train_metrics).items()})
            row.update({f"val_{k}": v for k, v in self._scalar_metrics(val_metrics).items()})
            row["time_s"] = round(time.time() - started, 1)

            if self.monitor not in row:
                raise KeyError(
                    f"Monitor '{self.monitor}' not found in the epoch metrics. "
                    f"Available: {[k for k in row if k not in ('epoch', 'lr', 'time_s')]}"
                )

            score = float(row[self.monitor])
            improved = self.early_stopping.step(score)

            if improved:
                self.best_metrics = val_metrics
                self._save_checkpoint(epoch, score, val_metrics)
                self.best_epoch = epoch
                flag = " *"
            else:
                flag = ""

            self.history.append(row)
            self._save_history()

            self.logger.info(
                "Epoch %03d/%03d | train loss %.4f acc %.4f f1 %.4f | "
                "val loss %.4f acc %.4f f1 %.4f qwk %.4f | %s=%.4f%s | %.1fs",
                epoch,
                self.epochs,
                row.get("train_loss", float("nan")),
                row.get("train_accuracy", float("nan")),
                row.get("train_f1_macro", float("nan")),
                row.get("val_loss", float("nan")),
                row.get("val_accuracy", float("nan")),
                row.get("val_f1_macro", float("nan")),
                row.get("val_qwk", float("nan")),
                self.monitor,
                score,
                flag,
                row["time_s"],
            )

            if self.early_stopping.should_stop:
                self.logger.info(
                    "Early stopping after %d epochs without improvement (best=%.4f).",
                    self.early_stopping.patience,
                    self.early_stopping.best,
                )
                break

        self._save_history()
        return pd.DataFrame(self.history)

    def predict(self, loader: DataLoader, return_meta: bool = False):
        """Return ``(y_true, y_pred)`` or ``(y_true, y_pred, meta)`` for a loader.

        ``meta`` is a list of dicts (``image_path``, ``patient_id``) when the
        dataset was created with ``return_meta=True``.
        """
        return predict_loader(
            self.model, loader, device=self.device, return_meta=return_meta
        )

    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        """Full metric dict (accuracy, F1, QWK, sensitivity, specificity...)."""
        y_true, y_pred = self.predict(loader)
        return compute_metrics(
            y_true,
            y_pred,
            num_classes=self.num_classes,
            class_names=self.class_names,
        )
