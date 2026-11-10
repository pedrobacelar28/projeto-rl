"""EpiCare: a procedurally generated longitudinal-treatment POMDP.

The agent observes 8 symptom intensities in ``[0, 1]``; the disease state is
hidden and exported only as a diagnostic ``state_id``. All 16 treatments are
admissible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from epicare import EpiCare as _EpiCare
from epicare.policies import Random, StandardOfCare

from projeto_rl.envs import register_env
from projeto_rl.envs.base import EnvAdapter, EnvSpec, FeatureSpec
from projeto_rl.policies import Policy, greedy

_REFERENCE_POLICIES = {"random": Random, "soc": StandardOfCare}


@register_env
class EpiCare(EnvAdapter):
    env_id = "epicare"
    behavior_policy_ids = {"random": 10, "soc": 12}

    def __init__(self, *, seed: int | None = None, environment_seed: int = 1, **kwargs):
        self.environment_seed = int(environment_seed)
        self.env = _EpiCare(seed=self.environment_seed, **kwargs)
        observation, _ = self.env.reset(seed=seed)
        num_actions = int(self.env.action_space.n)
        symptom_dim = int(np.asarray(observation).shape[0])
        self.spec = EnvSpec(
            env_id=self.env_id,
            num_actions=num_actions,
            features=tuple(FeatureSpec(f"symptom_{idx}", "numeric") for idx in range(symptom_dim)),
            action_labels=tuple(f"Treatment {idx}" for idx in range(num_actions)),
            metric_names=("symptom_burden", "remission", "adverse_event", "time_to_remission"),
            primary_metric="symptom_burden",
            num_states=int(self.env.n_diseases + 1),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any], *, seed: int | None = None, purpose: str) -> "EpiCare":
        del purpose  # every episode samples a fresh patient
        return cls(
            seed=seed, environment_seed=int(config.get("environment_seed", 1)), **dict(config.get("env_kwargs") or {})
        )

    def _state_id(self) -> int:
        if self.env.current_disease == "Remission":
            return int(self.env.n_diseases)
        return int(self.env.current_disease_index)

    def _info(self, observation: np.ndarray, info: dict) -> dict:
        enriched = dict(info)
        enriched["state_id"] = self._state_id()
        enriched["action_mask"] = np.ones(self.num_actions, dtype=np.float32)
        enriched["admissible_actions"] = list(range(self.num_actions))
        enriched.setdefault("remission", False)
        enriched.setdefault("adverse_event", False)
        enriched["symptom_burden"] = float(np.mean(observation))
        enriched["time_to_remission"] = (
            -1.0 if self.env.time_to_remission is None else float(self.env.time_to_remission)
        )
        return enriched

    def reset(self, seed: int | None = None):
        observation, info = self.env.reset(seed=seed)
        observation = np.asarray(observation, dtype=np.float32)
        return observation, self._info(observation, info)

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(int(action))
        observation = np.asarray(observation, dtype=np.float32)
        return observation, float(reward), bool(terminated), bool(truncated), self._info(observation, info)

    def close(self) -> None:
        self.env.close()

    def transition_metrics(self, info: dict) -> dict[str, float]:
        return {
            "symptom_burden": float(info["symptom_burden"]),
            "remission": float(bool(info.get("remission", False))),
            "adverse_event": float(bool(info.get("adverse_event", False))),
            "time_to_remission": float(info.get("time_to_remission", -1.0)),
        }

    def episode_metrics(
        self, *, total_reward: float, final_info: dict, transition_metrics: list[dict[str, float]]
    ) -> dict[str, float | bool]:
        del total_reward
        remission = bool(any(row.get("remission", 0.0) > 0.0 for row in transition_metrics))
        adverse_event = bool(any(row.get("adverse_event", 0.0) > 0.0 for row in transition_metrics))
        time_to_remission = float(final_info.get("time_to_remission", -1.0))
        return {
            "success": remission,
            "remission": remission,
            "adverse_event": adverse_event,
            "time_to_remission": time_to_remission if remission else np.nan,
        }

    def make_policy(self, name: str, *, seed: int = 0, options: dict[str, Any] | None = None) -> Policy:
        if name not in _REFERENCE_POLICIES:
            return super().make_policy(name, seed=seed, options=options)
        return EpiCareReference(_REFERENCE_POLICIES[name], self, seed=seed)


class EpiCareReference(Policy):
    """Adapter for the upstream EpiCare policies (Random, SoC).

    SoC adapts to rewards within an episode, so :meth:`observe` must be called
    after every step.
    """

    def __init__(self, factory, env: EpiCare, *, seed: int = 0):
        super().__init__(seed=seed)
        self.factory = factory
        self.bind_env(env)

    def bind_env(self, env: EpiCare) -> None:
        self.policy = self.factory(env.env)

    def reset_episode(self) -> None:
        self.policy.reset()

    def observe(self, action: int, reward: float) -> None:
        self.policy.update(int(action), float(reward))

    def action_probabilities(self, obs, info, deterministic=True):
        probabilities = np.asarray(self.policy.get_treatment_probs(obs), dtype=np.float64)
        probabilities /= probabilities.sum()
        return greedy(probabilities) if deterministic else probabilities
