"""Grid2Op adapter: columns, reward against the source's L2RPNReward, legality mask, episode ends, rule, and split."""

from __future__ import annotations

import grid2op
import numpy as np
from conftest import TINY_DATASETS
from grid2op.Reward import L2RPNReward
from lightsim2grid import LightSimBackend

from projeto_rl.envs import make_env
from projeto_rl.envs.grid2op import FEATURES, Rule

FEATURE_INDEX = {feature.name: index for index, feature in enumerate(FEATURES)}


def _env(purpose="evaluate"):
    return make_env(TINY_DATASETS["grid2op"], purpose=purpose)


def _random_admissible_rollout(env, rng, steps):
    """Yield ``(obs, info)`` while acting uniformly over the admissible actions."""

    obs, info = env.reset(seed=int(rng.integers(1000)))
    for _ in range(steps):
        yield obs, info
        obs, _, terminated, truncated, info = env.step(int(rng.choice(info["admissible_actions"])))
        if terminated or truncated:
            obs, info = env.reset(seed=int(rng.integers(1000)))


def test_columns_are_the_source_attributes():
    env = _env()
    obs, info = env.reset(seed=3)
    source = info["grid_observation"]
    assert len(FEATURES) == 153 and len(set(FEATURE_INDEX)) == 153
    assert obs[FEATURE_INDEX["hour_of_day"]] == source.hour_of_day
    assert obs[FEATURE_INDEX["minute_of_hour"]] == source.minute_of_hour / 5
    assert obs[FEATURE_INDEX["rho_7"]] == np.float32(source.rho[7])
    assert obs[FEATURE_INDEX["load_v_2"]] == np.float32(source.load_v[2])
    assert obs[FEATURE_INDEX["gen_theta_4"]] == np.float32(source.gen_theta[4])
    assert obs[FEATURE_INDEX["topo_vect_51"]] == source.topo_vect[51]
    # A disconnected element (bus -1) is code 0.
    rng = np.random.default_rng(0)
    for obs, info in _random_admissible_rollout(env, rng, 400):
        disconnected = info["grid_observation"].topo_vect[51] == -1
        if disconnected:
            assert obs[FEATURE_INDEX["topo_vect_51"]] == 0 and obs[FEATURE_INDEX["line_status_13"]] == 0
            break
    else:
        raise AssertionError("line 13 never disconnected")


def test_reward_is_the_source_l2rpn_reward_over_the_lines():
    env = _env()
    source = grid2op.make(
        TINY_DATASETS["grid2op"]["env_kwargs"]["dataset"], backend=LightSimBackend(), reward_class=L2RPNReward
    )
    _, info = env.reset(seed=5)
    options = {"time serie id": env.scenario, "max step": env.horizon, "init ts": env.start}
    source.reset(options=options if env.start > 0 else {k: v for k, v in options.items() if k != "init ts"})
    # The greedy rule at threshold 0 changes the topology at almost every step.
    rule = Rule(threshold=0.0)
    for _ in range(8):
        action = rule.act(None, info)
        _, reward, terminated, _, info = env.step(action)
        _, source_reward, done, _ = source.step(env.actions[action])
        assert not terminated and not done
        assert np.isclose(reward * env.source.n_line, source_reward, rtol=1e-5)
    source.close()


def test_mask_is_what_the_source_rules_accept():
    env = _env()
    rng = np.random.default_rng(1)
    refused = 0
    for _, info in _random_admissible_rollout(env, rng, 60):
        grid_obs = info["grid_observation"]
        for index, action in enumerate(env.actions):
            _, _, _, simulated = grid_obs.simulate(action)
            assert bool(info["action_mask"][index]) == (not simulated["is_illegal"]), index
            refused += int(simulated["is_illegal"])
    assert refused > 0


def test_blackout_is_terminal_and_the_day_is_truncated():
    env = _env()
    rng = np.random.default_rng(2)
    _, info = env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(int(rng.choice(info["admissible_actions"])))
    assert terminated and not truncated and reward == 0.0 and info["admissible_actions"] == [0]
    assert env.episode_metrics(total_reward=0.0, final_info=info, transition_metrics=[]) == {"success": False}
    env.reset(seed=0)
    for step in range(1, env.horizon + 1):
        _, _, terminated, truncated, info = env.step(0)
        assert not terminated and truncated == (step == env.horizon)
    assert env.episode_metrics(total_reward=0.0, final_info=info, transition_metrics=[]) == {"success": True}


def test_rule_takes_the_lowest_simulated_loading():
    env = _env()
    obs, info = env.reset(seed=7)
    assert Rule(threshold=10.0).act(obs, info) == 0
    loadings = np.full(env.num_actions, np.inf)
    for index in np.flatnonzero(info["action_mask"]):
        simulated, _, done, simulated_info = info["grid_observation"].simulate(env.actions[index])
        if not (done or simulated_info["exception"]):
            loadings[index] = simulated.rho.max()
    best = int(np.argmin(loadings))
    assert Rule(threshold=0.0).act(obs, info) == (best if loadings[best] < loadings[0] else 0)


def test_scenarios_are_split():
    evaluate, collect = _env(), _env("collect")
    evaluate = {evaluate.reset(seed=seed)[1]["state_id"] for seed in range(20)}
    collect = {collect.reset(seed=seed)[1]["state_id"] for seed in range(20)}
    assert collect <= set(range(0, 50)) and evaluate <= set(range(50, 100))
