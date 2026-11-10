"""The epsilon mixture used by every behavior policy."""

from __future__ import annotations

import numpy as np
import pytest

from projeto_rl.policies import EpsilonMixture, Policy


class _Fixed(Policy):
    def __init__(self, probabilities):
        super().__init__()
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.events = []

    def reset_episode(self):
        self.events.append("reset")

    def observe(self, action, reward):
        self.events.append((action, reward))

    def action_probabilities(self, obs, info, deterministic=True):
        return self.probabilities


def test_mixture_matches_the_formula_over_admissible_actions():
    inner = _Fixed([0.0, 1.0, 0.0, 0.0])
    mask = np.array([1.0, 1.0, 0.0, 1.0])
    mixed = EpsilonMixture(inner, 0.3).action_probabilities(None, {"action_mask": mask}, deterministic=False)
    np.testing.assert_allclose(mixed, [0.1, 0.7 + 0.1, 0.0, 0.1])


def test_mixture_forwards_episode_hooks_and_validates_epsilon():
    inner = _Fixed([1.0, 0.0])
    policy = EpsilonMixture(inner, 0.1)
    policy.reset_episode()
    policy.observe(1, 2.5)
    assert inner.events == ["reset", (1, 2.5)]
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonMixture(inner, 1.5)
