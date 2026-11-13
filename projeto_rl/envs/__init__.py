"""Environment registry.

An environment is one module that defines an :class:`EnvAdapter` subclass
decorated with :func:`register_env`, and is imported at the bottom of this
file. Nothing else in the package branches on environment ids.
"""

from __future__ import annotations

from typing import Any

from projeto_rl.envs.base import PURPOSES, EnvAdapter, EnvSpec, FeatureSpec

_ENVIRONMENTS: dict[str, type[EnvAdapter]] = {}
_ALIASES: dict[str, str] = {}


def register_env(cls: type[EnvAdapter]) -> type[EnvAdapter]:
    if cls.env_id in _ENVIRONMENTS:
        raise ValueError(f"Environment {cls.env_id!r} is already registered")
    _ENVIRONMENTS[cls.env_id] = cls
    for name in (cls.env_id, *cls.aliases):
        _ALIASES[name] = cls.env_id
    return cls


def environment_id(value: str | dict[str, Any] | None) -> str:
    raw = value.get("env_id") if isinstance(value, dict) else value
    if raw is None:
        raise ValueError("Missing env_id")
    try:
        return _ALIASES[str(raw).lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown env_id={raw!r}; registered: {environment_ids()}") from exc


def environment_ids() -> list[str]:
    return sorted(_ENVIRONMENTS)


def get_environment(env_id: str) -> type[EnvAdapter]:
    return _ENVIRONMENTS[environment_id(env_id)]


def make_env(config: dict[str, Any], *, seed: int | None = None, purpose: str) -> EnvAdapter:
    """Instantiate the environment described by ``config`` (``env_id``, ``env_kwargs``, ...).

    ``purpose`` is ``"collect"`` for dataset collection and ``"evaluate"`` for
    simulator evaluation; environments with a fixed entity pool keep the two
    on disjoint entities.
    """

    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
    return get_environment(environment_id(config)).from_config(config, seed=seed, purpose=purpose)


# Built-in environments register themselves on import.
from projeto_rl.envs import (  # noqa: E402,F401
    citylearn,
    epicare,
    grid2op,
    industrial_benchmark,
    sales_promotion,
    simbank,
)

__all__ = [
    "EnvAdapter",
    "EnvSpec",
    "FeatureSpec",
    "environment_id",
    "environment_ids",
    "get_environment",
    "make_env",
    "register_env",
]
