"""Unit tests for the patient-level splitting logic (data leakage guard)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.splitters import (
    assert_no_leakage,
    describe_splits,
    load_splits,
    patient_level_split,
    save_splits,
)


@pytest.fixture()
def brset_like_frame() -> pd.DataFrame:
    """40 patients x 2 eyes, 5 balanced grades -> 80 rows."""
    rows = []
    for patient in range(40):
        label = patient % 5
        for eye in ("left", "right"):
            rows.append(
                {
                    "image_path": f"P{patient:03d}_{eye}.jpg",
                    "patient_id": f"P{patient:03d}",
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def test_split_is_patient_level(brset_like_frame: pd.DataFrame) -> None:
    train, val, test = patient_level_split(brset_like_frame, seed=42)

    assert len(train) + len(val) + len(test) == len(brset_like_frame)
    assert set(train["patient_id"]).isdisjoint(val["patient_id"])
    assert set(train["patient_id"]).isdisjoint(test["patient_id"])
    assert set(val["patient_id"]).isdisjoint(test["patient_id"])
    assert_no_leakage(train, val, test)


def test_split_respects_requested_ratios(brset_like_frame: pd.DataFrame) -> None:
    train, val, test = patient_level_split(
        brset_like_frame, val_size=0.2, test_size=0.2, seed=7
    )
    n_patients = brset_like_frame["patient_id"].nunique()

    assert train["patient_id"].nunique() == pytest.approx(n_patients * 0.6, abs=2)
    assert val["patient_id"].nunique() == pytest.approx(n_patients * 0.2, abs=2)
    assert test["patient_id"].nunique() == pytest.approx(n_patients * 0.2, abs=2)


def test_split_is_reproducible(brset_like_frame: pd.DataFrame) -> None:
    first = patient_level_split(brset_like_frame, seed=123)
    second = patient_level_split(brset_like_frame, seed=123)
    for left, right in zip(first, second):
        pd.testing.assert_frame_equal(left, right)


def test_different_seeds_give_different_splits(brset_like_frame: pd.DataFrame) -> None:
    _, val_a, _ = patient_level_split(brset_like_frame, seed=1)
    _, val_b, _ = patient_level_split(brset_like_frame, seed=2)
    assert set(val_a["patient_id"]) != set(val_b["patient_id"])


def test_assert_no_leakage_detects_a_shared_patient(brset_like_frame: pd.DataFrame) -> None:
    train, val, test = patient_level_split(brset_like_frame, seed=42)
    leaked_patient = val["patient_id"].iloc[0]

    contaminated = pd.concat(
        [train, brset_like_frame[brset_like_frame["patient_id"] == leaked_patient]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(contaminated, val, test)


def test_invalid_ratios_are_rejected(brset_like_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        patient_level_split(brset_like_frame, val_size=0.6, test_size=0.5)
    with pytest.raises(ValueError):
        patient_level_split(brset_like_frame, val_size=-0.1)


def test_save_and_load_roundtrip(brset_like_frame: pd.DataFrame, tmp_path) -> None:
    splits = patient_level_split(brset_like_frame, seed=42)
    frames = {"train": splits[0], "val": splits[1], "test": splits[2]}

    created = save_splits(frames, out_dir=tmp_path)
    assert len(created) == 3
    for path in created.values():
        assert path.is_file()

    reloaded = load_splits(tmp_path)
    assert set(reloaded) == {"train", "val", "test"}
    pd.testing.assert_frame_equal(reloaded["train"], frames["train"])


def test_describe_splits_summary(brset_like_frame: pd.DataFrame) -> None:
    train, val, test = patient_level_split(brset_like_frame, seed=42)
    summary = describe_splits({"train": train, "val": val, "test": test})

    assert list(summary.index) == ["train", "val", "test"]
    assert summary.loc["train", "images"] == len(train)
    assert summary["patients"].sum() == brset_like_frame["patient_id"].nunique()
