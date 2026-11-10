"""SimBank adapter against the vendored simulator's online mode."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import SIMBANK_KWARGS, TINY_DATASETS
from simbank.simulation import PresProcessGenerator

from projeto_rl.envs import make_env
from projeto_rl.envs.simbank import FEATURE_INDEX, BankPolicy, GuideRandom, SimBank, _dataset_params


def _env(purpose="evaluate"):
    return make_env(TINY_DATASETS["simbank"], purpose=purpose)


def _direct_case(seed: int, actions: list[int]):
    """The guide notebook's online loop, called on a fresh generator."""

    generator = PresProcessGenerator(_dataset_params(SIMBANK_KWARGS), SIMBANK_KWARGS["base_seeds"]["evaluate"])
    generator.start_simulation_inference(seed_to_add=seed)
    taken = 0
    while generator.int_points_available:
        generator.continue_simulation_inference(actions[taken])
        taken += 1
    return taken, generator.end_simulation_inference()


def test_adapter_reproduces_the_online_loop_and_pays_the_outcome_at_the_end():
    env = _env()
    actions = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    for seed in (0, 3, 11):
        env.reset(seed=seed)
        rewards = []
        done = False
        while not done:
            _, reward, terminated, truncated, info = env.step(actions[len(rewards)])
            rewards.append(reward)
            done = terminated or truncated
        taken, case = _direct_case(seed, actions)
        if taken == 0:
            continue  # the adapter skips cases without a decision point
        assert len(rewards) == taken
        assert rewards[:-1] == [0.0] * (taken - 1)
        assert rewards[-1] == pytest.approx(case[-1]["outcome"] / SIMBANK_KWARGS["outcome_unit"])
        assert info["outcome"] == case[-1]["outcome"]


def test_reset_restores_the_base_seed():
    env = _env()
    first = env.reset(seed=4)[0]
    env.reset(seed=9)
    np.testing.assert_array_equal(env.reset(seed=4)[0], first)


def test_collection_and_evaluation_use_the_two_guide_seeds():
    assert _env("collect").base_seed == 82 and _env("evaluate").base_seed == 130


def test_bank_rule_matches_the_guide():
    policy = BankPolicy(SIMBANK_KWARGS["policies_info"])
    info = {"action_mask": np.ones(2)}

    def obs(**values):
        vector = np.zeros(len(FEATURE_INDEX), dtype=np.float32)
        for name, value in values.items():
            vector[FEATURE_INDEX[name]] = value
        return vector

    assert policy.act(obs(amount=20000, est_quality=5, unc_quality=0, noc=1), info) == 1  # contact
    assert policy.act(obs(amount=9000, est_quality=5, unc_quality=0, noc=1), info) == 0  # too small
    assert policy.act(obs(amount=20000, est_quality=1, unc_quality=0, noc=3), info) == 0  # cancel condition
    assert policy.act(obs(amount=20000, est_quality=5, unc_quality=2, noc=0), info) == 0  # still uncertain


def test_guide_random_acts_once_and_keeps_its_action():
    policy = GuideRandom(depth=4, width=2, seed=0)
    info = {"action_mask": np.ones(2)}
    for _ in range(20):
        policy.reset_episode()
        assert policy.timing in (0, 2, 4, 6)
        for position in range(8):
            probabilities = policy.action_probabilities(None, info, deterministic=False)
            if position < policy.timing:
                np.testing.assert_array_equal(probabilities, [1.0, 0.0])
            elif position == policy.timing:
                np.testing.assert_array_equal(probabilities, [0.5, 0.5])
                chosen = int(policy.rng.integers(2))
            else:
                np.testing.assert_array_equal(probabilities, np.eye(2)[chosen])
            policy.observe(chosen if position == policy.timing else int(np.argmax(probabilities)), 0.0)


def test_only_time_contact_hq_is_supported():
    kwargs = {**SIMBANK_KWARGS, "intervention": {**SIMBANK_KWARGS["intervention"], "name": "choose_procedure"}}
    kwargs.pop("base_seeds")
    with pytest.raises(ValueError, match="time_contact_HQ"):
        SimBank(base_seed=130, **kwargs)
