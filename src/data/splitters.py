"""Patient level train/val/test splitting to prevent data leakage.

BRSET contains two eyes per patient (often several visits). A naive random split
puts the left eye of a patient in train and the right eye in test; because both
eyes share the same patient, the model leaks information and the test score
becomes optimistic. Every split below therefore groups by ``patient_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from src.utils.config import resolve_project_path

__all__ = [
    "patient_level_split",
    "assert_no_leakage",
    "save_splits",
    "load_splits",
    "describe_splits",
]


def _patient_labels(
    frame: pd.DataFrame, patient_column: str, label_column: str
) -> pd.DataFrame:
    """One row per patient with its majority label (used for stratification)."""
    grouped = frame.groupby(patient_column, sort=True)[label_column].agg(
        lambda values: values.value_counts().idxmax()
    )
    return grouped.reset_index().rename(columns={label_column: "patient_label"})


def _check_sizes(val_size: float, test_size: float) -> None:
    if not 0 <= val_size < 1 or not 0 <= test_size < 1:
        raise ValueError("val_size and test_size must be in [0, 1).")
    if val_size + test_size >= 1:
        raise ValueError(
            f"val_size ({val_size}) + test_size ({test_size}) must be < 1 "
            "so that a non-empty train split remains."
        )


def patient_level_split(
    frame: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    patient_column: str = "patient_id",
    label_column: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe so that no patient appears in two splits.

    Stratification is done on the *patient* majority label; if a class has too
    few patients for stratification the function falls back to a grouped random
    split instead of failing.

    Returns
    -------
    (train_df, val_df, test_df)
        Dataframes with the original columns, index reset.
    """
    if patient_column not in frame.columns:
        raise KeyError(f"Column '{patient_column}' not found; cannot group by patient.")
    if label_column not in frame.columns:
        raise KeyError(f"Column '{label_column}' not found.")
    _check_sizes(val_size, test_size)

    patients = _patient_labels(frame, patient_column, label_column)
    labels = patients["patient_label"].to_numpy()
    groups = patients[patient_column].to_numpy()

    # --- test split ---------------------------------------------------------
    if test_size > 0:
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            rest_idx, test_idx = next(splitter.split(patients, labels))
        except ValueError:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            rest_idx, test_idx = next(splitter.split(patients, groups=groups))
    else:
        rest_idx = np.arange(len(patients))
        test_idx = np.array([], dtype=int)

    # --- validation split (from the remaining patients) ---------------------
    relative_val = val_size / max(1e-9, (1.0 - test_size))
    rest = patients.iloc[rest_idx]
    if val_size > 0 and len(rest) > 1:
        try:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=relative_val, random_state=seed
            )
            train_idx, val_idx = next(
                splitter.split(rest, rest["patient_label"].to_numpy())
            )
        except ValueError:
            splitter = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
            train_idx, val_idx = next(
                splitter.split(rest, groups=rest[patient_column].to_numpy())
            )
        train_patients = set(rest.iloc[train_idx][patient_column])
        val_patients = set(rest.iloc[val_idx][patient_column])
    else:
        train_patients = set(rest[patient_column])
        val_patients = set()

    test_patients = set(patients.iloc[test_idx][patient_column])

    def _subset(patient_set: Iterable[Any]) -> pd.DataFrame:
        mask = frame[patient_column].isin(set(patient_set))
        return frame.loc[mask].reset_index(drop=True)

    return _subset(train_patients), _subset(val_patients), _subset(test_patients)


def _overlap_report(
    splits: Mapping[str, pd.DataFrame], patient_column: str
) -> Dict[str, set]:
    sets = {
        name: set(df[patient_column].astype(str)) if patient_column in df.columns else set()
        for name, df in splits.items()
    }
    names = list(sets)
    overlaps: Dict[str, set] = {}
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = sets[first] & sets[second]
            if shared:
                overlaps[f"{first}&{second}"] = shared
    return overlaps


def assert_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    patient_column: str = "patient_id",
    n_examples: int = 5,
) -> bool:
    """Raise ``ValueError`` if a patient shows up in more than one split.

    Returns ``True`` when the split is clean, which makes it usable as an
    assertion inside tests and notebooks.
    """
    splits = {"train": train_df, "val": val_df}
    if test_df is not None:
        splits["test"] = test_df

    overlaps = _overlap_report(splits, patient_column)
    if overlaps:
        details = "; ".join(
            f"{pair}: {len(shared)} patient(s) e.g. {sorted(shared)[:n_examples]}"
            for pair, shared in overlaps.items()
        )
        raise ValueError(f"Patient-level data leakage detected -> {details}")
    return True


def save_splits(
    splits: Mapping[str, pd.DataFrame],
    out_dir: str | Path = "data/splits",
    filenames: Mapping[str, str] | None = None,
) -> Dict[str, Path]:
    """Write ``train/val/test`` dataframes next to ``.gitkeep`` in ``data/splits``."""
    target_dir = resolve_project_path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    filenames = dict(filenames or {})

    created: Dict[str, Path] = {}
    for name, frame in splits.items():
        filename = filenames.get(name, f"{name}.csv")
        path = target_dir / filename
        frame.to_csv(path, index=False)
        created[name] = path
    return created


def load_splits(
    splits_dir: str | Path = "data/splits",
    filenames: Mapping[str, str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Read the split CSVs written by :func:`save_splits`."""
    source_dir = resolve_project_path(splits_dir)
    filenames = dict(filenames or {"train": "train.csv", "val": "val.csv", "test": "test.csv"})

    splits: Dict[str, pd.DataFrame] = {}
    for name, filename in filenames.items():
        path = source_dir / filename
        if path.is_file():
            splits[name] = pd.read_csv(path)
    if not splits:
        raise FileNotFoundError(f"No split CSV found in {source_dir}.")
    return splits


def describe_splits(
    splits: Mapping[str, pd.DataFrame],
    patient_column: str = "patient_id",
    label_column: str = "label",
    class_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summary table (images, patients, per-class counts) for the report."""
    rows: Dict[str, Dict[str, Any]] = {}
    for name, frame in splits.items():
        row: Dict[str, Any] = {"split": name, "images": len(frame)}
        row["patients"] = (
            int(frame[patient_column].nunique()) if patient_column in frame.columns else np.nan
        )
        if label_column in frame.columns:
            counts = frame[label_column].value_counts().sort_index()
            for label, count in counts.items():
                column = (
                    f"class_{class_names[int(label)]}"
                    if class_names is not None and int(label) < len(class_names)
                    else f"class_{label}"
                )
                row[column] = int(count)
        rows[name] = row
    return pd.DataFrame(rows.values()).set_index("split")


def make_stratified_patient_splits(
    frame: pd.DataFrame,
    out_dir: str | Path = "data/splits",
    **kwargs: Any,
) -> Dict[str, pd.DataFrame]:
    """Split, validate and persist in one call (convenience for notebooks)."""
    val_size = kwargs.pop("val_size", 0.15)
    test_size = kwargs.pop("test_size", 0.15)
    seed = kwargs.pop("seed", 42)
    patient_column = kwargs.pop("patient_column", "patient_id")
    label_column = kwargs.pop("label_column", "label")

    train_df, val_df, test_df = patient_level_split(
        frame,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        patient_column=patient_column,
        label_column=label_column,
    )
    assert_no_leakage(train_df, val_df, test_df, patient_column=patient_column)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    save_splits(splits, out_dir=out_dir)
    return splits
