"""``torch.utils.data.Dataset`` for the BRSET diabetic retinopathy task.

The dataset reads a split CSV (``image_path``, ``patient_id``, ``label``) and
returns normalised ImageNet tensors. Preprocessing (ROI / CLAHE / mask) is
applied online through the ``preprocess`` callable, so the ``data/processed``
folder is optional: you can train directly on the raw JPGs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.utils.config import PROJECT_ROOT, resolve_project_path

try:  # albumentations is the preferred backend
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    _HAS_ALBUMENTATIONS = True
except ImportError:  # pragma: no cover - fallback for minimal environments
    A = None  # type: ignore[assignment]
    ToTensorV2 = None  # type: ignore[assignment]
    _HAS_ALBUMENTATIONS = False

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_CLASS_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")

__all__ = [
    "DRDataset",
    "build_transforms",
    "build_dataloaders",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DEFAULT_CLASS_NAMES",
]


def build_transforms(image_size: int = 384, train: bool = False) -> Callable:
    """Return the augmentation pipeline for train or evaluation.

    Uses ``albumentations`` when installed, otherwise falls back to
    ``torchvision.transforms`` so the project still imports on a bare install.
    """
    if _HAS_ALBUMENTATIONS:
        if train:
            return A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3
                    ),
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]
            )
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )

    from torchvision import transforms as T

    if train:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.15, contrast=0.15),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _is_albumentations_transform(transform: Callable) -> bool:
    module = type(transform).__module__
    return _HAS_ALBUMENTATIONS and module.startswith("albumentations")


class DRDataset(Dataset):
    """Retinal image dataset backed by a split CSV.

    Parameters
    ----------
    source:
        Path to a CSV file or a ``pandas.DataFrame`` already in memory.
    image_dir:
        Folder that contains the images. Used to resolve relative paths.
    transform:
        ``albumentations`` or ``torchvision`` transform. ``None`` builds the
        standard pipeline from :func:`build_transforms`.
    image_size:
        Used when ``transform`` is ``None`` (384 by default).
    train:
        Selects the augmentation pipeline when ``transform`` is ``None``.
    label_column / path_column / patient_column:
        Column names inside the CSV.
    label_map:
        Optional mapping from raw label values to ``0..K-1``; handy when the CSV
        stores label names instead of integers.
    preprocess:
        Optional callable ``(np.ndarray) -> np.ndarray`` applied before the
        transform (typically ``src.preprocessing.preprocess_image``).
    return_meta:
        Return a dict with ``image``, ``label``, ``patient_id`` and
        ``image_path`` (used by the error analysis and the Grad-CAM notebook).
    """

    def __init__(
        self,
        source: str | Path | pd.DataFrame,
        image_dir: str | Path | None = None,
        transform: Callable | None = None,
        image_size: int = 384,
        train: bool = False,
        label_column: str = "label",
        path_column: str = "image_path",
        patient_column: str = "patient_id",
        class_names: Sequence[str] | None = None,
        label_map: Mapping[Any, int] | None = None,
        preprocess: Callable[[np.ndarray], np.ndarray] | None = None,
        return_meta: bool = False,
        check_files: bool = False,
    ) -> None:
        if isinstance(source, (str, Path)):
            csv_path = resolve_project_path(source)
            if not csv_path.is_file():
                raise FileNotFoundError(f"Split CSV not found: {csv_path}")
            frame = pd.read_csv(csv_path)
        elif isinstance(source, pd.DataFrame):
            frame = source.copy()
        else:
            raise TypeError("source must be a CSV path or a pandas DataFrame.")

        if frame.empty:
            raise ValueError("The split is empty: nothing to load.")

        frame = frame.reset_index(drop=True)
        for column in (path_column, label_column):
            if column not in frame.columns:
                raise KeyError(
                    f"Column '{column}' missing from the split. "
                    f"Available columns: {list(frame.columns)}"
                )

        self.frame = frame
        self.image_dir = resolve_project_path(image_dir) if image_dir else None
        self.image_size = int(image_size)
        self.label_column = label_column
        self.path_column = path_column
        self.patient_column = patient_column
        self.class_names = list(class_names) if class_names else list(DEFAULT_CLASS_NAMES)
        self.preprocess = preprocess
        self.return_meta = bool(return_meta)

        self.labels = np.asarray(self._encode_labels(frame[label_column], label_map), dtype=np.int64)
        self.image_paths = [self._resolve_image_path(p) for p in frame[path_column]]

        if check_files:
            missing = [str(p) for p in self.image_paths if not Path(p).is_file()]
            if missing:
                preview = "\n".join(missing[:5])
                raise FileNotFoundError(
                    f"{len(missing)} image(s) referenced by the split do not exist. "
                    f"First entries:\n{preview}"
                )

        self.transform = transform if transform is not None else build_transforms(
            self.image_size, train=train
        )
        self._use_albumentations = _is_albumentations_transform(self.transform)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _encode_labels(values: Iterable[Any], label_map: Mapping[Any, int] | None) -> list[int]:
        encoded: list[int] = []
        for value in values:
            if label_map is not None:
                try:
                    encoded.append(int(label_map[value]))
                except KeyError as exc:
                    raise KeyError(f"Label {value!r} not found in label_map.") from exc
                continue
            try:
                encoded.append(int(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Label {value!r} is not an integer. Pass `label_map` to encode "
                    "string labels (e.g. {'No DR': 0, 'Mild': 1, ...})."
                ) from exc
        return encoded

    def _resolve_image_path(self, value: Any) -> Path:
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path

        candidates = []
        if self.image_dir is not None:
            candidates.append(self.image_dir / path)
        candidates.extend(
            [
                PROJECT_ROOT / "data" / "processed" / path,
                PROJECT_ROOT / "data" / "raw" / path,
                PROJECT_ROOT / path,
            ]
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return candidates[0]

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        if cv2 is not None:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            raise FileNotFoundError(f"Unable to read image: {path}")
        with Image.open(path) as handle:
            return np.asarray(handle.convert("RGB"))

    def _apply_transform(self, image: np.ndarray) -> torch.Tensor:
        if self._use_albumentations:
            tensor = self.transform(image=image)["image"]
        else:
            tensor = self.transform(Image.fromarray(image))
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        return tensor.float()

    # --------------------------------------------------------------- dataset API
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Any:
        image = self._read_image(Path(self.image_paths[index]))

        if image.ndim == 2 or image.shape[-1] == 4:
            if cv2 is not None:
                image = (
                    cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    if image.ndim == 2
                    else cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                )
            else:
                image = np.asarray(Image.fromarray(image).convert("RGB"))

        if self.preprocess is not None:
            image = self.preprocess(image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        tensor = self._apply_transform(image)
        label = int(self.labels[index])

        if not self.return_meta:
            return tensor, label

        return {
            "image": tensor,
            "label": label,
            "patient_id": self._patient_id(index),
            "image_path": str(self.image_paths[index]),
        }

    # ------------------------------------------------------------------ metadata
    def _patient_id(self, index: int) -> Any:
        if self.patient_column in self.frame.columns:
            return self.frame.iloc[index][self.patient_column]
        return None

    @property
    def num_classes(self) -> int:
        return int(max(len(self.class_names), int(self.labels.max()) + 1 if len(self) else 1))

    @property
    def targets(self) -> np.ndarray:
        """Integer labels ordered like the dataset (useful for class weights)."""
        return self.labels.copy()

    def class_distribution(self) -> pd.Series:
        counts = np.bincount(self.labels, minlength=self.num_classes)
        return pd.Series(
            counts,
            index=[self.class_names[i] if i < len(self.class_names) else str(i) for i in range(len(counts))],
            name="images",
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetics
        return (
            f"{self.__class__.__name__}(n={len(self)}, classes={self.num_classes}, "
            f"image_size={self.image_size}, transform={type(self.transform).__name__})"
        )


def build_dataloaders(
    train_source: str | Path | pd.DataFrame,
    val_source: str | Path | pd.DataFrame,
    test_source: str | Path | pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    image_dir: str | Path | None = None,
    preprocess: Callable[[np.ndarray], np.ndarray] | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    image_size: int | None = None,
    pin_memory: bool | None = None,
) -> Dict[str, DataLoader]:
    """Create ``train`` / ``val`` (and optional ``test``) data loaders.

    Reads defaults from ``model_config.yaml`` (``training``) and
    ``data_config.yaml`` (``image.size``) when ``config`` is given.
    """
    config = config or {}
    training_cfg = dict(config.get("training", {}) if config else {})
    data_image_cfg = dict((config.get("image", {}) or {}) if config else {})
    split_cfg = dict((config.get("splits", {}) or {}) if config else {})

    batch_size = int(batch_size or training_cfg.get("batch_size", 32))
    image_size = int(image_size or data_image_cfg.get("size", 384))

    if num_workers is None:
        num_workers = int(training_cfg.get("num_workers", 4))
        if os.name == "nt" and num_workers > 0:
            # Windows spawns processes per worker; notebooks usually prefer 0.
            num_workers = 0
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dataset_kwargs = dict(
        image_dir=image_dir,
        image_size=image_size,
        label_column=split_cfg.get("label_column", "label"),
        path_column=split_cfg.get("image_path_column", "image_path"),
        patient_column=split_cfg.get("patient_column", "patient_id"),
        class_names=(config.get("classes", {}) or {}).get("names"),
        preprocess=preprocess,
    )

    loaders: Dict[str, DataLoader] = {}
    for name, source, is_train in (
        ("train", train_source, True),
        ("val", val_source, False),
        ("test", test_source, False),
    ):
        if source is None:
            continue
        dataset = DRDataset(source=source, train=is_train, **dataset_kwargs)
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )
    return loaders
