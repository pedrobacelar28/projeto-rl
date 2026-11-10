"""SalesPromotion adapter: per-user episodes, observable-state update, and user split."""

from __future__ import annotations

import numpy as np
from conftest import TINY_DATASETS

from projeto_rl.envs import make_env
from projeto_rl.envs.sales_promotion import ACTIONS, TicketBand, action_index


def _env(purpose="evaluate"):
    return make_env(TINY_DATASETS["sales_promotion"], purpose=purpose)


def test_actions_are_coupon_count_times_discount():
    assert len(ACTIONS) == 48
    assert ACTIONS[0] == (0, 0.95) and ACTIONS[-1] == (5, 0.6)
    assert action_index(3, 0.85) == 3 * 8 + 2


def test_observable_update_follows_get_next_state(monkeypatch):
    env = _env()
    obs, _ = env.reset(seed=2)
    total, average_orders, average_fee, weekday, day = (float(value) for value in obs)
    # The user model answers 2 orders at 30.0 each, coupon cost 4.0, GMV 56.0.
    monkeypatch.setattr(
        env.market, "step", lambda action: (4.0, 56.0, np.array([[2.0, 30.0]]), np.array([4.0]), np.array([56.0]))
    )
    next_obs, reward, terminated, _, info = env.step(0)
    expected = [
        total + 2.0,
        average_orders + (2.0 - average_orders) / 61,
        average_fee + (30.0 - average_fee) / 61,
        (weekday + 1) % 7,
        day + 1,
    ]
    np.testing.assert_allclose(next_obs, expected, rtol=1e-6)
    assert reward == 56.0 - 4.0 and info["daily_orders"] == 2.0 and not terminated


def test_orders_and_fee_are_zeroed_together(monkeypatch):
    env = _env()
    obs, _ = env.reset(seed=2)
    monkeypatch.setattr(
        env.market, "step", lambda action: (0.0, 0.0, np.array([[3.0, 0.0]]), np.array([0.0]), np.array([0.0]))
    )
    next_obs, _, _, _, info = env.step(0)
    assert next_obs[0] == obs[0] and info["daily_orders"] == 0.0


def test_episode_lasts_the_horizon_and_users_are_split():
    env = _env()
    _, info = env.reset(seed=0)
    steps = 0
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(5)
        steps += 1
        done = terminated or truncated
    assert steps == TINY_DATASETS["sales_promotion"]["env_kwargs"]["horizon"]
    collect = {_env("collect").reset(seed=seed)[1]["state_id"] for seed in range(200)}
    evaluate = {env.reset(seed=seed)[1]["state_id"] for seed in range(200)}
    assert collect <= set(range(0, 50)) and evaluate <= set(range(50, 100))


def test_per_user_rewards_sum_to_the_source_total():
    env = _env()
    env.reset(seed=1)
    env.market.val_initial_states = env.internal_states[:20].copy()
    env.market.reset()
    actions = np.array([[coupons, discount] for coupons, discount in ACTIONS[:20]], dtype=np.float64)
    total_cost, total_gmv, _, cost, gmv = env.market.step(actions)
    assert np.isclose(cost.sum(), total_cost) and np.isclose(gmv.sum(), total_gmv)


def test_ticket_band_picks_the_offer_of_the_fee_band():
    policy = TicketBand(thresholds=[1.8, 12.0], band_actions=[[5, 0.7], [3, 0.85], [1, 0.95]])
    info = {"action_mask": np.ones(48)}
    for fee, offer in ((0.5, (5, 0.7)), (5.0, (3, 0.85)), (30.0, (1, 0.95))):
        assert policy.act(np.array([0, 0, fee, 1, 0], dtype=np.float32), info) == action_index(*offer)
