"""The contract every state representation implements, and the policy wrapper that applies it.

A representation maps raw tabular observations ``[N, d]`` to vectors
``[N, k]`` before any algorithm sees them. It is fitted once on the train
split, frozen, stored in the checkpoint, and applied unchanged to validation
data and to simulator observations at evaluation time.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, ClassVar

import numpy as np

from projeto_rl.envs.base import FeatureSpec
from projeto_rl.policies import Policy


class Representation:
    #: Registry key, stored as ``representation`` in run configs.
    name: ClassVar[str]

    def __init__(self, features: tuple[FeatureSpec, ...], config: dict[str, Any]):
        if not features:
            raise ValueError(f"Representation {self.name!r} needs at least one feature")
        self.features = tuple(features)
        self.config = config

    @property
    def input_dim(self) -> int:
        return len(self.features)

    # -- implemented by each representation ------------------------------
    def fit(self, train, *, seed: int) -> None:
        """Fit on the train split (an :class:`OfflineTransitionDataset`) only."""

        raise NotImplementedError

    def _transform(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        raise NotImplementedError

    def fitted_state(self) -> dict[str, Any]:
        """Everything :meth:`fit` learned, as JSON-friendly values or arrays."""

        raise NotImplementedError

    def load_fitted_state(self, state: dict[str, Any]) -> None:
        raise NotImplementedError

    # -- shared -----------------------------------------------------------
    def transform(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != self.input_dim:
            raise ValueError(
                f"Representation {self.name!r} expects observations of shape [N, {self.input_dim}], got {obs.shape}"
            )
        out = np.asarray(self._transform(obs), dtype=np.float32)
        if out.shape != (obs.shape[0], self.output_dim):
            raise RuntimeError(
                f"Representation {self.name!r} returned shape {out.shape}, expected [N, {self.output_dim}]"
            )
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "features": [asdict(feature) for feature in self.features],
            "fitted": self.fitted_state(),
        }


class RepresentationPolicy(Policy):
    """Deploys a policy trained on represented observations in the raw-observation simulator."""

    def __init__(self, policy: Policy, representation: Representation):
        super().__init__()
        self.policy = policy
        self.representation = representation

    @property
    def stateless(self) -> bool:
        return self.policy.stateless

    def _represent(self, obs) -> np.ndarray:
        return self.representation.transform(np.asarray(obs, dtype=np.float32)[None, :])[0]

    def bind_env(self, env) -> None:
        self.policy.bind_env(env)

    def reset_episode(self) -> None:
        self.policy.reset_episode()

    def observe(self, action: int, reward: float) -> None:
        self.policy.observe(action, reward)

    def action_probabilities(self, obs, info, deterministic=True):
        return self.policy.action_probabilities(self._represent(obs), info, deterministic=deterministic)

    def act(self, obs, info, deterministic=True) -> int:
        return self.policy.act(self._represent(obs), info, deterministic=deterministic)

    def act_batch(self, observations, infos, deterministic=True) -> list[int]:
        # One transform for all the episodes: a learned representation costs
        # about the same for one observation as for a batch of them.
        represented = self.representation.transform(np.asarray(observations, dtype=np.float32))
        return self.policy.act_batch(represented, infos, deterministic=deterministic)
