"""Representation registry.

A representation is one module that defines a :class:`Representation`
subclass decorated with :func:`register_representation`, and is imported at
the bottom of this file. Runs pick one with the ``representation`` config key.
"""

from __future__ import annotations

from typing import Any

from projeto_rl.envs.base import FeatureSpec
from projeto_rl.representations.base import Representation, RepresentationPolicy

_REPRESENTATIONS: dict[str, type[Representation]] = {}


def register_representation(cls: type[Representation]) -> type[Representation]:
    if cls.name in _REPRESENTATIONS:
        raise ValueError(f"Representation {cls.name!r} is already registered")
    _REPRESENTATIONS[cls.name] = cls
    return cls


def representation_names() -> list[str]:
    return sorted(_REPRESENTATIONS)


def get_representation(name: str) -> type[Representation]:
    try:
        return _REPRESENTATIONS[str(name)]
    except KeyError as exc:
        raise KeyError(f"Unknown representation {name!r}; registered: {representation_names()}") from exc


def build_representation(config: dict[str, Any], features: tuple[FeatureSpec, ...]) -> Representation:
    """An unfitted instance of ``config['representation']``."""

    return get_representation(config["representation"])(features, config)


def load_representation(config: dict[str, Any], state: dict[str, Any]) -> Representation:
    """Rebuild a fitted representation from :meth:`Representation.state_dict` (no refitting)."""

    if state["name"] != config["representation"]:
        raise ValueError(f"Checkpoint representation {state['name']!r} != config {config['representation']!r}")
    features = tuple(FeatureSpec.from_dict(item) for item in state["features"])
    representation = build_representation(config, features)
    representation.load_fitted_state(state["fitted"])
    return representation


# Built-in representations register themselves on import.
from projeto_rl.representations import raw, spr, standardize, tabpfn, tdjepa, tjepa  # noqa: E402,F401

__all__ = [
    "Representation",
    "RepresentationPolicy",
    "build_representation",
    "get_representation",
    "load_representation",
    "register_representation",
    "representation_names",
]
