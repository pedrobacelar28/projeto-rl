"""Industrial Benchmark adapter against the vendored simulator."""

from __future__ import annotations

import numpy as np
from conftest import TINY_DATASETS
from industrial_benchmark_python.IDS import IDS

from projeto_rl.envs import make_env
from projeto_rl.envs.industrial_benchmark import ACTIONS, Mediocre, action_index


def test_actions_are_the_ibgym_discrete_grid():
    # IBGym(action_type="discrete") enumerates velocity, then gain, then shift over (-1, 0, 1).
    expected = [[v, g, s] for v in (-1, 0, 1) for g in (-1, 0, 1) for s in (-1, 0, 1)]
    assert [list(action) for action in ACTIONS] == expected
    assert action_index(0, 0, 0) == 13 and action_index(-1, -1, -1) == 0 and action_index(1, 1, 1) == 26


def test_adapter_reproduces_the_vendored_simulator():
    env = make_env(TINY_DATASETS["industrial_benchmark"], purpose="evaluate")
    obs, _ = env.reset(seed=5)
    rng = np.random.default_rng(5)
    reference = IDS(rng, p=int(rng.choice([30, 70])))
    assert obs[0] == reference.state["p"]
    actions = [0, 26, 13, 7, 19, 3, 22, 13, 13, 1, 25, 9]
    for step, action in enumerate(actions, start=1):
        obs, reward, terminated, truncated, _ = env.step(action)
        reference.step(np.asarray(ACTIONS[action], dtype=np.float64))
        np.testing.assert_allclose(obs, [reference.state[key] for key in ("p", "v", "g", "h", "f", "c")], rtol=1e-6)
        assert reward == -reference.state["cost"] and not terminated
        assert truncated == (step == 12)


def test_mediocre_moves_each_lever_toward_the_target():
    policy = Mediocre(target=50, deadband=5)
    info = {"action_mask": np.ones(27)}
    obs = np.array([70, 40, 56, 52, 0, 0], dtype=np.float32)  # velocity low, gain high, shift inside the band
    assert policy.act(obs, info) == action_index(1, -1, 0)
