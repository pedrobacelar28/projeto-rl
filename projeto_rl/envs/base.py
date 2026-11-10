"""The environment contract every benchmark adapter implements."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import numpy as np

from projeto_rl.policies import Policy

FEATURE_KINDS = ("numeric", "binary", "ordinal", "categorical", "cyclic")
# Kinds whose values are integer codes ``0..cardinality-1``.
CODED_KINDS = ("ordinal", "categorical", "cyclic")
PURPOSES = ("collect", "evaluate")


@dataclass(frozen=True)
class FeatureSpec:
    """Name and type of one observation column, read by representations."""

    name: str
    kind: str
    cardinality: int | None = None

    def __post_init__(self):
        if self.kind not in FEATURE_KINDS:
            raise ValueError(f"Feature {self.name!r} has kind {self.kind!r}; expected one of {FEATURE_KINDS}")
        if self.kind in CODED_KINDS and (self.cardinality is None or int(self.cardinality) < 2):
            raise ValueError(f"Feature {self.name!r} of kind {self.kind!r} needs a cardinality >= 2")
        if self.kind not in CODED_KINDS and self.cardinality is not None:
            raise ValueError(f"Feature {self.name!r} of kind {self.kind!r} takes no cardinality")

    def check(self, values: np.ndarray) -> None:
        """Raise if ``values`` (one column) violate the declared type."""

        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Feature {self.name!r} has non-finite values")
        if self.kind == "binary" and not np.isin(values, (0.0, 1.0)).all():
            raise ValueError(f"Binary feature {self.name!r} has values outside {{0, 1}}")
        if self.kind in CODED_KINDS:
            codes_ok = np.equal(values, np.round(values)) & (values >= 0) & (values < int(self.cardinality))
            if not codes_ok.all():
                raise ValueError(f"Feature {self.name!r} has values outside the codes 0..{int(self.cardinality) - 1}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureSpec":
        return cls(name=str(payload["name"]), kind=str(payload["kind"]), cardinality=payload.get("cardinality"))


@dataclass(frozen=True)
class EnvSpec:
    """Static description persisted with every dataset and evaluation."""

    env_id: str
    num_actions: int
    features: tuple[FeatureSpec, ...]
    action_labels: tuple[str, ...]
    metric_names: tuple[str, ...] = ()
    primary_metric: str | None = None
    action_geometry: dict[str, Any] = field(default_factory=lambda: {"type": "categorical"})
    # Size of the simulator's diagnostic state space, when it has one. The
    # state id is logged for analysis only and is never an agent input.
    num_states: int | None = None

    @property
    def obs_dim(self) -> int:
        return len(self.features)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("action_labels", "metric_names"):
            payload[key] = list(payload[key])
        payload["features"] = [asdict(feature) for feature in self.features]
        payload["obs_dim"] = self.obs_dim
        return payload


class EnvAdapter:
    """Gymnasium-style environment with the benchmark's semantic surface.

    ``reset``/``step`` return a ``float32`` observation vector and an ``info``
    dict that always contains ``action_mask`` (``float32[num_actions]``),
    ``admissible_actions`` (list of ints), and ``state_id`` (diagnostic int).

    Subclasses are registered with :func:`projeto_rl.envs.register_env` and
    declare their behavior and baseline policies in ``behavior_policy_ids``.
    """

    env_id: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]] = ()
    # Stable integer ids written to ``behavior_policy_ids`` in datasets.
    behavior_policy_ids: ClassVar[dict[str, int]] = {}

    spec: EnvSpec

    @classmethod
    def from_config(cls, config: dict[str, Any], *, seed: int | None = None, purpose: str) -> "EnvAdapter":
        """Build the environment; ``purpose`` is ``"collect"`` (datasets) or ``"evaluate"`` (simulator scores)."""

        raise NotImplementedError

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def make_policy(self, name: str, *, seed: int = 0, options: dict[str, Any] | None = None) -> Policy:
        """Build one of the environment's declared behavior/reference policies."""

        raise ValueError(
            f"{self.spec.env_id} declares no policy {name!r}; available: {sorted(self.behavior_policy_ids)}"
        )

    @property
    def num_actions(self) -> int:
        return self.spec.num_actions

    @property
    def obs_dim(self) -> int:
        return self.spec.obs_dim

    def action_label(self, action: int) -> str:
        labels = self.spec.action_labels
        return labels[int(action)] if 0 <= int(action) < len(labels) else str(int(action))

    def action_distance(self, previous: int, current: int) -> float:
        return float(int(previous) != int(current))

    def action_is_abrupt(self, previous: int, current: int) -> bool:
        return bool(self.action_distance(previous, current) > 0.0)

    def transition_metrics(self, info: dict) -> dict[str, float]:
        """Per-step metrics, ordered as ``spec.metric_names``."""

        return {}

    def episode_metrics(
        self, *, total_reward: float, final_info: dict, transition_metrics: list[dict[str, float]]
    ) -> dict[str, float | bool]:
        """Episode outcomes. Must contain ``success``."""

        del final_info, transition_metrics
        return {"success": bool(total_reward > 0.0)}

    def export_columns(self, info: dict, action: int) -> dict[str, float | int]:
        """Extra environment-specific columns for exported trajectories."""

        del info, action
        return {}
