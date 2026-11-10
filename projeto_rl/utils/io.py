"""File IO helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(data, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_pickle(data, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as handle:
        pickle.dump(data, handle)


def load_pickle(path: str | Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)
