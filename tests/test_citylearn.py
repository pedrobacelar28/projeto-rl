"""CityLearn adapter: observation against the dataset files, the hour's bill, episodes, home split, and the schedule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import TINY_DATASETS

from projeto_rl.envs import make_env
from projeto_rl.envs.citylearn import FEATURE_INDEX, TimeOfUse


def _env(purpose="evaluate"):
    return make_env(TINY_DATASETS["citylearn"], purpose=purpose)


def test_observation_holds_the_dataset_row_of_the_episode_start():
    env = _env()
    obs, info = env.reset(seed=4)
    root = env.source.root_directory
    row = pd.read_csv(f"{root}/Building_{info['state_id']}.csv").iloc[env.start]
    weather = pd.read_csv(f"{root}/weather.csv").iloc[env.start]
    pricing = pd.read_csv(f"{root}/pricing.csv").iloc[env.start]
    # Calendar codes start at 0 here and at 1 in the source.
    assert obs[FEATURE_INDEX["month"]] == row["month"] - 1
    assert obs[FEATURE_INDEX["hour"]] == row["hour"] - 1
    assert obs[FEATURE_INDEX["day_type"]] == row["day_type"] - 1
    assert np.isclose(obs[FEATURE_INDEX["non_shiftable_load"]], row["non_shiftable_load"], rtol=1e-6)
    # The file holds W per kW of panel.
    pv_kw = env.source.buildings[0].pv.nominal_power
    assert np.isclose(obs[FEATURE_INDEX["solar_generation"]], row["solar_generation"] * pv_kw / 1000.0, rtol=1e-5)
    assert np.isclose(obs[FEATURE_INDEX["outdoor_dry_bulb_temperature"]], weather["outdoor_dry_bulb_temperature"])
    assert np.isclose(obs[FEATURE_INDEX["electricity_pricing"]], pricing["electricity_pricing"])
    assert obs[FEATURE_INDEX["electrical_storage_soc"]] == 0.0


def test_reward_is_the_bill_of_the_hour():
    env = _env()
    obs, _ = env.reset(seed=4)
    load, solar, price = (
        float(obs[FEATURE_INDEX[name]]) for name in ("non_shiftable_load", "solar_generation", "electricity_pricing")
    )
    # From an empty battery, +0.2 of the 5 kW nominal power draws exactly 1 kWh from the grid.
    charge = int(np.flatnonzero(np.isclose(env.levels, 0.2))[0])
    next_obs, reward, _, _, _ = env.step(charge)
    net = load - solar + 1.0
    assert np.isclose(next_obs[FEATURE_INDEX["net_electricity_consumption"]], net, atol=1e-5)
    assert np.isclose(reward, -price * max(net, 0.0), atol=1e-5)


def test_exported_energy_earns_nothing():
    env = _env()
    for seed in range(200):
        obs, _ = env.reset(seed=seed)
        if obs[FEATURE_INDEX["solar_generation"]] > obs[FEATURE_INDEX["non_shiftable_load"]] + 0.5:
            break
    else:
        pytest.fail("no episode starts with a solar surplus")
    idle = int(np.flatnonzero(np.isclose(env.levels, 0.0))[0])
    next_obs, reward, _, _, _ = env.step(idle)
    assert next_obs[FEATURE_INDEX["net_electricity_consumption"]] < 0.0 and reward == 0.0


def test_episode_lasts_the_horizon_and_homes_are_split():
    env = _env()
    env.reset(seed=0)
    steps, terminated, truncated = 0, False, False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(5)
        steps += 1
    assert steps == TINY_DATASETS["citylearn"]["env_kwargs"]["horizon"] and truncated and not terminated
    collect = {_env("collect").reset(seed=seed)[1]["state_id"] for seed in range(30)}
    evaluate = {env.reset(seed=seed)[1]["state_id"] for seed in range(30)}
    assert collect == {1, 2} and evaluate == {3}


def test_time_of_use_follows_the_schedule():
    levels = np.linspace(-1.0, 1.0, 11)
    policy = TimeOfUse(
        levels=levels, charge_hours=[10, 11], charge_level=0.2, discharge_hours=[17], discharge_level=-0.4
    )
    info = {"action_mask": np.ones(11)}
    obs = np.zeros(len(FEATURE_INDEX), dtype=np.float32)
    # Hours ending 10, 17, and 3 are the codes 9, 16, and 2.
    for code, level in ((9, 0.2), (16, -0.4), (2, 0.0)):
        obs[FEATURE_INDEX["hour"]] = code
        assert np.isclose(levels[policy.act(obs, info)], level)
    with pytest.raises(ValueError, match="not one of the battery levels"):
        TimeOfUse(levels=levels, charge_hours=[10], charge_level=0.25, discharge_hours=[17], discharge_level=-0.4)
