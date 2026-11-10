"""NeoRL SalesPromotion: a coupon-allocation simulator built from a learned model of real users.

Qin et al., "NeoRL: A Near Real-World Benchmark for Offline Reinforcement
Learning", NeurIPS 2022 Datasets and Benchmarks; user model, initial states,
and transition logic vendored from the NeoRL ``benchmark`` branch, Apache-2.0
(see third_party/sales-promotion/NOTICE).

One episode is one user over the 30 evaluation days of the source
(``SalesPromotion_v0``). Each day the platform sends ``coupons`` coupons
(0-5) at one discount level (0.95 ... 0.60, the source's
``DISCOUNT_COUPON_LIST``): 6 x 8 = 48 actions. The observation is the
source's four columns (total orders, average orders per day, average fee per
day, weekday), updated by ``SalesPromotion_v0._get_next_state``, plus the
episode day. The reward is the source's ``gmv - cost`` (``per_gmv`` already
subtracts the coupon cost, and the source subtracts it again), per user.

Adaptations: the source steps all 10 000 users together and returns only the
summed reward; here each episode is one user, with the user's own reward.
The 10 000 users are split once: ``collect_users`` for datasets and
``evaluate_users`` for simulator scores, so evaluation never sees a user of
the training data. The user model keeps an 18-column internal state, so the
task is partially observable, as in the source.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import numpy as np
import torch
from sales_promotion.marketing import MarketingEnv
from sales_promotion.models import EnsembleVenvModel, VenvPolicy

from projeto_rl.envs import register_env
from projeto_rl.envs.base import EnvAdapter, EnvSpec, FeatureSpec
from projeto_rl.policies import Policy, greedy, masked_uniform, one_hot

DISCOUNTS = tuple(MarketingEnv.DISCOUNT_COUPON_LIST)
COUPONS = tuple(range(MarketingEnv.MAX_DELIVER_NUMBER + 1))
ACTIONS = tuple((coupons, discount) for coupons in COUPONS for discount in DISCOUNTS)
FEATURES = (
    FeatureSpec("total_orders", "numeric"),
    FeatureSpec("avg_orders", "numeric"),
    FeatureSpec("avg_fee", "numeric"),
    FeatureSpec("weekday", "cyclic", 7),
    FeatureSpec("day", "numeric"),
)
# SalesPromotion_v0.reset: the evaluation starts on day 61, after 60 days of history.
HISTORY_DAYS = 60
ENSEMBLE_SIZE = 5


def action_index(coupons: int, discount: float) -> int:
    return ACTIONS.index((int(coupons), float(discount)))


def _user_model() -> EnsembleVenvModel:
    model = EnsembleVenvModel([VenvPolicy() for _ in range(ENSEMBLE_SIZE)])
    state = torch.load(files("sales_promotion") / "assets" / "user_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()


@register_env
class SalesPromotion(EnvAdapter):
    env_id = "sales_promotion"
    aliases = ("sales-promotion",)
    behavior_policy_ids = {"random": 40, "ticket_band": 41}

    def __init__(self, *, users: list[int], horizon: int, seed: int | None = None):
        assets = files("sales_promotion") / "assets"
        self.internal_states = np.load(assets / "test_initial_states_10000_people.npy")
        self.initial_obs = np.load(assets / "evaluation_init_obs.npy").astype(np.float64)
        low, high = (int(value) for value in users)
        if not 0 <= low < high <= len(self.internal_states):
            raise ValueError(f"User range {users} is outside 0..{len(self.internal_states)}")
        self.users = (low, high)
        self.horizon = int(horizon)
        self.market = MarketingEnv(
            self.internal_states[low : low + 1].copy(),
            _user_model(),
            torch.device("cpu"),
            torch.Generator(),
            np.random.RandomState(0),
        )
        self.spec = EnvSpec(
            env_id=self.env_id,
            num_actions=len(ACTIONS),
            features=FEATURES,
            action_labels=tuple(f"{coupons} coupons at {discount:.2f}" for coupons, discount in ACTIONS),
            metric_names=("daily_orders",),
            primary_metric="daily_orders",
            num_states=len(self.internal_states),
        )
        self.reset(seed=seed)

    @classmethod
    def from_config(cls, config: dict[str, Any], *, seed: int | None = None, purpose: str) -> "SalesPromotion":
        kwargs = dict(config["env_kwargs"])
        users = kwargs.pop("users")[purpose]
        return cls(users=users, seed=seed, **kwargs)

    def _info(self, daily_orders: float) -> dict:
        return {
            "state_id": int(self.user),
            "action_mask": np.ones(self.num_actions, dtype=np.float32),
            "admissible_actions": list(range(self.num_actions)),
            "daily_orders": float(daily_orders),
        }

    def _observation(self) -> np.ndarray:
        return np.asarray([*self.obs, self.day], dtype=np.float32)

    def reset(self, seed: int | None = None):
        seed = 0 if seed is None else int(seed)
        low, high = self.users
        self.user = low + int(np.random.default_rng(seed).integers(high - low))
        self.market.val_initial_states = self.internal_states[self.user : self.user + 1].copy()
        self.market.torch_generator = torch.Generator().manual_seed(seed)
        self.market.np_rng = np.random.RandomState(seed % 2**32)
        self.market.reset()
        self.obs = self.initial_obs[self.user].copy()
        self.accumulated_days = HISTORY_DAYS
        self.day = 0
        return self._observation(), self._info(0.0)

    def step(self, action: int):
        coupons, discount = ACTIONS[int(action)]
        _, _, user_actions, cost, gmv = self.market.step(np.asarray([[coupons, discount]], dtype=np.float64))
        order_num, average_fee = (float(value) for value in user_actions[0])
        # SalesPromotion_v0._get_next_state, for one user.
        order_num = float(np.round(np.clip(order_num, 0, 6)))
        average_fee = float(np.clip(average_fee, 0.0, 100.0))
        if average_fee <= 0.0:
            order_num = 0.0
        if order_num <= 0:
            average_fee = 0.0
        self.accumulated_days += 1
        self.obs[0] += order_num
        self.obs[1] += (order_num - self.obs[1]) / self.accumulated_days
        self.obs[2] += (average_fee - self.obs[2]) / self.accumulated_days
        self.obs[3] = (self.obs[3] + 1) % 7
        self.day += 1
        reward = float(gmv[0] - cost[0])
        return self._observation(), reward, self.day >= self.horizon, False, self._info(order_num)

    def transition_metrics(self, info: dict) -> dict[str, float]:
        return {"daily_orders": float(info["daily_orders"])}

    def make_policy(self, name: str, *, seed: int = 0, options: dict[str, Any] | None = None) -> Policy:
        if name == "random":
            return UniformRandom(seed=seed)
        if name == "ticket_band":
            options = dict(options or {})
            return TicketBand(
                thresholds=options["ticket_band_thresholds"], band_actions=options["ticket_band_actions"], seed=seed
            )
        return super().make_policy(name, seed=seed, options=options)


class UniformRandom(Policy):
    """Uniform over the 48 coupon actions."""

    def action_probabilities(self, obs, info, deterministic=True):
        probabilities = masked_uniform(info["action_mask"])
        return greedy(probabilities) if deterministic else probabilities


class TicketBand(Policy):
    """A fixed coupon offer per band of the user's average fee (lower ticket, more and deeper coupons).

    ``thresholds`` split ``avg_fee`` into ``len(thresholds) + 1`` bands;
    ``band_actions`` gives ``[coupons, discount]`` for each band, lowest first.
    """

    def __init__(self, *, thresholds: list[float], band_actions: list[list[float]], seed: int = 0):
        super().__init__(seed=seed)
        if len(band_actions) != len(thresholds) + 1 or list(thresholds) != sorted(thresholds):
            raise ValueError("ticket_band needs sorted thresholds and one action per band")
        self.thresholds = [float(value) for value in thresholds]
        self.actions = [action_index(int(coupons), float(discount)) for coupons, discount in band_actions]

    def action_probabilities(self, obs, info, deterministic=True):
        band = int(np.searchsorted(self.thresholds, float(obs[2]), side="right"))
        return one_hot(len(ACTIONS), self.actions[band])
