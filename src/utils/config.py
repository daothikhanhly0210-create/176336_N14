"""Load the YAML configuration files that drive the project.

Every path inside the YAML files is written relative to the project root, so the
same configs work locally and on Google Colab right after ``git clone``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

DEFAULT_CONFIG_FILES = ("data_config.yaml", "model_config.yaml", "xai_config.yaml")

__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DEFAULT_CONFIG_FILES",
    "resolve_project_path",
    "load_yaml",
    "deep_merge",
    "load_configs",
    "load_project_configs",
]


def resolve_project_path(path: str | Path) -> Path:
    """Return an absolute path; relative paths are rooted at the project folder."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load one YAML file.

    ``path`` may be absolute, relative to the project root, or simply a file name
    that lives in ``configs/`` (e.g. ``"model_config.yaml"``).
    """
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        options = [PROJECT_ROOT / candidate, CONFIG_DIR / candidate]
        candidate = next((opt for opt in options if opt.is_file()), options[0])

    if not candidate.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path} (looked in {PROJECT_ROOT} and {CONFIG_DIR})"
        )

    with candidate.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file {candidate} must contain a YAML mapping, got {type(data)!r}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_configs(*names: str) -> Dict[str, Any]:
    """Load several configs and key them by file stem (``data``, ``model``, ``xai``)."""
    configs: Dict[str, Any] = {}
    for name in names or DEFAULT_CONFIG_FILES:
        configs[Path(name).stem.replace("_config", "")] = load_yaml(name)
    return configs


def load_project_configs() -> Dict[str, Any]:
    """Shorthand used by notebooks: ``{"data": ..., "model": ..., "xai": ...}``."""
    return load_configs(*DEFAULT_CONFIG_FILES)
