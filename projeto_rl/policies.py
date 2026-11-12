"""The policy contract shared by behavior policies, baselines, and learned agents.

A policy maps ``(obs, info)`` to a probability vector over the discrete
actions. Everything else (greedy or stochastic action selection and behavior
propensities for logged datasets) is derived from that single method, so a
policy written once can both collect data and be evaluated as a baseline.
"""

from __future__ import annotations

import math

import numpy as np


def masked_uniform(action_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(action_mask, dtype=np.float64)
    return mask / max(float(mask.sum()), 1.0)


def one_hot(num_actions: int, action: int) -> np.ndarray:
    probabilities = np.zeros(int(num_actions), dtype=np.float64)
    probabilities[int(action)] = 1.0
    return probabilities


def greedy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    return one_hot(len(probabilities), int(np.argmax(probabilities))).astype(probabilities.dtype, copy=False)


def sample_action(rng: np.random.Generator, probabilities: np.ndarray) -> tuple[int, float]:
    """Sample one action and return it with its exact log propensity."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    action = int(rng.choice(len(probabilities), p=probabilities))
    return action, float(math.log(max(probabilities[action], 1e-12)))


class Policy:
    """Base class for every policy in the benchmark.

    Subclasses implement :meth:`action_probabilities`. Stateful policies
    (recurrent agents, adaptive clinical policies) also override
    :meth:`reset_episode` and :meth:`observe`.
    """

    #: True when one instance can act in several episodes at once: it keeps
    #: no state between calls and reads nothing from the bound environment.
    stateless = False

    def __init__(self, *, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def bind_env(self, env) -> None:
        """Called by simulator evaluation with the environment it will act in.

        Only policies that read simulator internals (e.g. privileged oracles)
        need to override this.
        """

    def reset_episode(self) -> None:
        """Called before the first step of every episode."""

    def observe(self, action: int, reward: float) -> None:
        """Called after every environment step with the executed action."""

    def action_probabilities(self, obs, info: dict, deterministic: bool = True) -> np.ndarray:
        raise NotImplementedError

    def act(self, obs, info: dict, deterministic: bool = True) -> int:
        probabilities = self.action_probabilities(obs, info, deterministic=deterministic)
        if deterministic:
            return int(np.argmax(probabilities))
        return int(self.rng.choice(len(probabilities), p=np.asarray(probabilities, dtype=np.float64)))

    def act_batch(self, observations, infos, deterministic: bool = True) -> list[int]:
        """One action per ``(obs, info)`` pair; simulator evaluation passes one pair per running episode."""

        return [self.act(obs, info, deterministic=deterministic) for obs, info in zip(observations, infos, strict=True)]


class EpsilonMixture(Policy):
    """``(1 - epsilon) * policy + epsilon * uniform`` over the admissible actions.

    Episode hooks are forwarded, so adaptive policies (EpiCare's standard of
    care) keep seeing the actions and rewards of the episode.
    """

    def __init__(self, policy: Policy, epsilon: float, *, seed: int = 0):
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        super().__init__(seed=seed)
        self.policy = policy
        self.epsilon = float(epsilon)

    def bind_env(self, env) -> None:
        self.policy.bind_env(env)

    def reset_episode(self) -> None:
        self.policy.reset_episode()

    def observe(self, action: int, reward: float) -> None:
        self.policy.observe(action, reward)

    def action_probabilities(self, obs, info, deterministic=True):
        inner = np.asarray(self.policy.action_probabilities(obs, info, deterministic=False), dtype=np.float64)
        mixed = (1.0 - self.epsilon) * inner + self.epsilon * masked_uniform(info["action_mask"])
        return greedy(mixed) if deterministic else mixed
