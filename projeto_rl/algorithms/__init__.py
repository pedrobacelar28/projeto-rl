"""Algorithm registry.

An algorithm is one module that defines an :class:`OfflineAlgorithm` subclass
decorated with :func:`register_algorithm`, and is imported at the bottom of
this file. Its default hyperparameters live in ``configs/algorithms/<name>.json``.
"""

from __future__ import annotations

from typing import Any

from projeto_rl.algorithms.base import DataSpec, OfflineAlgorithm

_ALGORITHMS: dict[str, type[OfflineAlgorithm]] = {}


def register_algorithm(cls: type[OfflineAlgorithm]) -> type[OfflineAlgorithm]:
    for key in {cls.name, cls.method_id}:
        if key in _ALGORITHMS:
            raise ValueError(f"Algorithm {key!r} is already registered")
        _ALGORITHMS[key] = cls
    return cls


def get_algorithm(name: str) -> type[OfflineAlgorithm]:
    """Look an algorithm up by its ``algo`` name or public method id."""

    try:
        return _ALGORITHMS[str(name)]
    except KeyError as exc:
        raise KeyError(f"Unknown algorithm {name!r}; registered: {algorithm_names()}") from exc


def algorithm_names() -> list[str]:
    return sorted({cls.name for cls in _ALGORITHMS.values()})


def algorithms() -> list[type[OfflineAlgorithm]]:
    return [get_algorithm(name) for name in algorithm_names()]


def load_policy(config: dict[str, Any], checkpoint: dict[str, Any]):
    """Rebuild the deployed policy of a trained run: the frozen representation, then the algorithm's policy."""

    from projeto_rl.representations import RepresentationPolicy, load_representation
    from projeto_rl.utils.torch_utils import get_torch_device

    spec = DataSpec(**checkpoint["data_spec"])
    representation = load_representation(config, checkpoint["representation_state"])
    if representation.output_dim != spec.obs_dim:
        raise ValueError(
            f"Checkpoint representation outputs {representation.output_dim} != network input {spec.obs_dim}"
        )
    algorithm = get_algorithm(config["algo"]).from_checkpoint(config, spec, get_torch_device(), checkpoint)
    return RepresentationPolicy(algorithm.make_policy(), representation)


# Built-in algorithms register themselves on import.
from projeto_rl.algorithms import bc, cql, dqn, iql  # noqa: E402,F401

__all__ = [
    "DataSpec",
    "OfflineAlgorithm",
    "algorithm_names",
    "algorithms",
    "get_algorithm",
    "load_policy",
    "register_algorithm",
]
