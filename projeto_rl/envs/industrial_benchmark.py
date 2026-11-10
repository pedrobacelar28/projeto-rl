"""Industrial Benchmark: a stochastic industrial-control simulator with 27 discrete actions.

Hein et al., "A Benchmark Environment Motivated by Industrial Control
Problems", IEEE SSCI 2017 (arXiv:1709.09480); Python implementation vendored
from github.com/siemens/industrialbenchmark (see third_party/industrial-benchmark/NOTICE).

The agent observes the six current variables (setpoint, velocity, gain, shift,
fatigue, consumption), without the 30-step history window of the original
offline datasets, and each step moves velocity, gain, and shift by -1, 0, or
+1, the discrete action set of the source's ``IBGym(action_type="discrete")``.
Reward is the negative cost, as ``reward_type="classic"``.

Adaptations: each episode draws its setpoint from ``setpoints`` and is
truncated after ``horizon`` steps; ``state_id`` is the setpoint (diagnostic
only). The task has no success event, so ``success`` is always false.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
from industrial_benchmark_python.IDS import IDS

from projeto_rl.envs import register_env
from projeto_rl.envs.base import EnvAdapter, EnvSpec, FeatureSpec
from projeto_rl.policies import Policy, greedy, masked_uniform, one_hot

FEATURES = ("setpoint", "velocity", "gain", "shift", "fatigue", "consumption")
# Same order as IBGym: velocity varies slowest, shift fastest.
ACTIONS = tuple(product((-1, 0, 1), repeat=3))


def action_index(delta_velocity: int, delta_gain: int, delta_shift: int) -> int:
    return ACTIONS.index((int(delta_velocity), int(delta_gain), int(delta_shift)))


@register_env
class IndustrialBenchmark(EnvAdapter):
    env_id = "industrial_benchmark"
    aliases = ("industrial-benchmark",)
    behavior_policy_ids = {"random": 20, "mediocre": 21}

    def __init__(self, *, setpoints: list[int], horizon: int, seed: int | None = None):
        if not setpoints or int(horizon) <= 0:
            raise ValueError("industrial_benchmark needs a non-empty setpoints list and horizon > 0")
        self.setpoints = [int(value) for value in setpoints]
        self.horizon = int(horizon)
        self.spec = EnvSpec(
            env_id=self.env_id,
            num_actions=len(ACTIONS),
            features=tuple(FeatureSpec(name, "numeric") for name in FEATURES),
            action_labels=tuple(f"velocity {v:+d}, gain {g:+d}, shift {s:+d}" for v, g, s in ACTIONS),
            metric_names=("cost",),
            primary_metric="cost",
        )
        self.reset(seed=seed)

    @classmethod
    def from_config(cls, config: dict[str, Any], *, seed: int | None = None, purpose: str) -> "IndustrialBenchmark":
        del purpose  # every episode is a fresh draw of the simulator
        return cls(seed=seed, **dict(config["env_kwargs"]))

    def _observe(self) -> tuple[np.ndarray, dict]:
        obs = np.asarray([self.ids.state[key] for key in ("p", "v", "g", "h", "f", "c")], dtype=np.float32)
        info = {
            "state_id": int(self.ids.state["p"]),
            "action_mask": np.ones(self.num_actions, dtype=np.float32),
            "admissible_actions": list(range(self.num_actions)),
            "cost": float(self.ids.state["cost"]),
        }
        return obs, info

    def reset(self, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.ids = IDS(rng, p=int(rng.choice(self.setpoints)))
        self.steps = 0
        return self._observe()

    def step(self, action: int):
        self.ids.step(np.asarray(ACTIONS[int(action)], dtype=np.float64))
        self.steps += 1
        obs, info = self._observe()
        return obs, -float(self.ids.state["cost"]), False, self.steps >= self.horizon, info

    def transition_metrics(self, info: dict) -> dict[str, float]:
        return {"cost": float(info["cost"])}

    def episode_metrics(self, *, total_reward, final_info, transition_metrics):
        del total_reward, final_info, transition_metrics
        return {"success": False}

    def make_policy(self, name: str, *, seed: int = 0, options: dict[str, Any] | None = None) -> Policy:
        if name == "random":
            return UniformRandom(seed=seed)
        if name == "mediocre":
            options = dict(options or {})
            return Mediocre(
                target=float(options["mediocre_target"]), deadband=float(options["mediocre_deadband"]), seed=seed
            )
        return super().make_policy(name, seed=seed, options=options)


class UniformRandom(Policy):
    """Uniform over the admissible actions."""

    def action_probabilities(self, obs, info, deterministic=True):
        probabilities = masked_uniform(info["action_mask"])
        return greedy(probabilities) if deterministic else probabilities


class Mediocre(Policy):
    """A simple operator rule: move each of velocity, gain, and shift toward ``target``.

    The source repository ships no behavior policies (only stable-baselines
    training scripts), so this rule stands in for a mediocre human operator.
    A lever within ``deadband`` of the target is left unchanged.
    """

    def __init__(self, *, target: float, deadband: float, seed: int = 0):
        super().__init__(seed=seed)
        self.target = float(target)
        self.deadband = float(deadband)

    def action_probabilities(self, obs, info, deterministic=True):
        deltas = []
        for value in np.asarray(obs, dtype=np.float64)[1:4]:
            if value < self.target - self.deadband:
                deltas.append(1)
            elif value > self.target + self.deadband:
                deltas.append(-1)
            else:
                deltas.append(0)
        return one_hot(len(ACTIONS), action_index(*deltas))
