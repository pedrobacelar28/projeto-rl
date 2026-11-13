"""CityLearn: home-battery control on one year of measured load and solar generation from 17 homes.

Vázquez-Canteli, Kämpf, Henze, and Nagy, "CityLearn v1.0: An OpenAI Gym
Environment for Demand Response with Deep Reinforcement Learning", BuildSys
2019; Nweye, Sankaranarayanan, and Nagy, "The CityLearn Challenge 2022:
Overview, Results, and Lessons Learned", PMLR 220, 2023. Simulator: the
``citylearn`` package 3.2.0 (MIT), installed, not vendored; dataset
``citylearn_challenge_2022_phase_all``, downloaded by the package. Its loads
and solar generation are hourly data measured from 17 single-family homes in
Fontana, California (August 2016 to July 2017), cleaned by EPRI; weather,
carbon intensity, and the time-of-use tariff are shared by all homes.

One episode is one home over ``horizon`` hours starting at an hour drawn from
the year, with the battery empty, as every CityLearn reset leaves it. Each hour
the agent sets the battery to one of ``action_levels`` evenly spaced values in
[-1, 1], the source's ``electrical_storage`` action: the energy asked from the
battery as a fraction of its 5 kW nominal power (the battery model clips it to
what its power curve, capacity, and charge allow). The observation is the
source's 28 active observations of one building, in its order: calendar
(``month``, ``day_type``, ``hour``), outdoor temperature, humidity, diffuse and
direct solar irradiance with their 6, 12 and 24 h ahead values, carbon
intensity, the home's load, solar generation, battery state of charge, and net
consumption, and the electricity price with its 6, 12 and 24 h ahead values.
The reward is minus the electricity bill of the hour, the building's
``net_electricity_consumption_cost`` (price times the net consumption, with
exported energy earning nothing).

Adaptations and deviations:

- The source's reward functions ignore the price; the reward here is the
  source's cost KPI instead.
- The source's calendar codes start at 1 (``hour`` 1-24 is the hour ending,
  ``day_type`` 1-7 is Monday-Sunday, 8 a holiday that this dataset never
  has); the observation holds them minus one, as 0-based cyclic codes.
- As in the source, the state of charge and the net consumption in an
  observation are those of the previous hour.
- The source ends an episode one hour early and repeats the last observation
  as the terminal one; the adapter runs the source for ``horizon + 2`` hours
  and truncates after ``horizon`` steps, so every next observation is a real
  one.
- ``random_seed`` draws the battery's efficiency and power curves, so it is
  fixed by the config instead of following the episode seed.
- Homes are split once: ``collect`` homes for datasets and ``evaluate`` homes
  for simulator scores, so evaluation never sees a home of the training data.
- The source's rule-based controllers (``BasicRBC``, ``OptimizedRBC``) were
  written for an action that was a share of capacity and do worse than an
  idle battery here; the behavior policy is the time-of-use rule below.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from citylearn.citylearn import CityLearnEnv

from projeto_rl.envs import register_env
from projeto_rl.envs.base import EnvAdapter, EnvSpec, FeatureSpec
from projeto_rl.policies import Policy, greedy, masked_uniform, one_hot

FORECASTS = ("", "_predicted_1", "_predicted_2", "_predicted_3")
FEATURES = (
    FeatureSpec("month", "cyclic", 12),
    FeatureSpec("day_type", "cyclic", 7),
    FeatureSpec("hour", "cyclic", 24),
    *(
        FeatureSpec(f"{name}{suffix}", "numeric")
        for name in (
            "outdoor_dry_bulb_temperature",
            "outdoor_relative_humidity",
            "diffuse_solar_irradiance",
            "direct_solar_irradiance",
        )
        for suffix in FORECASTS
    ),
    *(
        FeatureSpec(name, "numeric")
        for name in (
            "carbon_intensity",
            "non_shiftable_load",
            "solar_generation",
            "electrical_storage_soc",
            "net_electricity_consumption",
        )
    ),
    *(FeatureSpec(f"electricity_pricing{suffix}", "numeric") for suffix in FORECASTS),
)
FEATURE_INDEX = {feature.name: index for index, feature in enumerate(FEATURES)}
# Calendar observations the source codes from 1.
ONE_BASED = ("month", "day_type", "hour")
# Hours the source runs beyond the horizon: its last hour is cut, and its last
# observation repeats the one before.
SOURCE_EXTRA_HOURS = 2


@register_env
class CityLearn(EnvAdapter):
    env_id = "citylearn"
    behavior_policy_ids = {"random": 50, "time_of_use": 51}

    def __init__(
        self,
        *,
        dataset: str,
        buildings: list[int],
        horizon: int,
        action_levels: int,
        random_seed: int,
        seed: int | None = None,
    ):
        if int(horizon) <= 0 or int(action_levels) < 2 or not buildings:
            raise ValueError("citylearn needs horizon > 0, action_levels >= 2, and at least one building")
        self.dataset = str(dataset)
        self.buildings = [int(building) for building in buildings]
        self.horizon = int(horizon)
        self.random_seed = int(random_seed)
        self.levels = np.linspace(-1.0, 1.0, int(action_levels))
        # One source environment per home, built on first use (about 50 ms each).
        self.sources: dict[int, CityLearnEnv] = {}
        self.spec = EnvSpec(
            env_id=self.env_id,
            num_actions=len(self.levels),
            features=FEATURES,
            action_labels=tuple(f"battery {level:+.1f} x nominal power" for level in self.levels),
            metric_names=("net_consumption",),
            primary_metric="net_consumption",
        )
        self.reset(seed=seed)

    @classmethod
    def from_config(cls, config: dict[str, Any], *, seed: int | None = None, purpose: str) -> "CityLearn":
        kwargs = dict(config["env_kwargs"])
        buildings = kwargs.pop("buildings")[purpose]
        return cls(buildings=buildings, seed=seed, **kwargs)

    def _source(self, building: int) -> CityLearnEnv:
        if building not in self.sources:
            source = CityLearnEnv(
                self.dataset,
                central_agent=True,
                buildings=[f"Building_{building}"],
                episode_time_steps=self.horizon + SOURCE_EXTRA_HOURS,
                rolling_episode_split=True,
                random_seed=self.random_seed,
            )
            names = tuple(source.observation_names[0])
            if names != tuple(feature.name for feature in FEATURES):
                raise RuntimeError(f"CityLearn observations of Building_{building} changed: {names}")
            self.sources[building] = source
        return self.sources[building]

    def _observe(self, source_obs) -> tuple[np.ndarray, dict]:
        obs = np.asarray(source_obs[0], dtype=np.float64)
        for name in ONE_BASED:
            obs[FEATURE_INDEX[name]] -= 1.0
        if obs[FEATURE_INDEX["day_type"]] >= 7:
            raise RuntimeError("CityLearn reported a holiday day_type; the observation codes only the 7 weekdays")
        info = {
            "state_id": int(self.building),
            "action_mask": np.ones(self.num_actions, dtype=np.float32),
            "admissible_actions": list(range(self.num_actions)),
            "net_consumption": float(obs[FEATURE_INDEX["net_electricity_consumption"]]),
        }
        return obs.astype(np.float32), info

    def reset(self, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.building = int(rng.choice(self.buildings))
        self.source = self._source(self.building)
        tracker = self.source.episode_tracker
        last_start = tracker.simulation_time_steps - (self.horizon + SOURCE_EXTRA_HOURS)
        self.start = int(rng.integers(last_start + 1))
        source_obs, _ = self.source.reset(options={"episode_index": self.start})
        if tracker.episode_start_time_step != self.start:
            raise RuntimeError(f"CityLearn started at hour {tracker.episode_start_time_step}, not {self.start}")
        self.steps = 0
        return self._observe(source_obs)

    def step(self, action: int):
        source_obs, _, terminated, truncated, _ = self.source.step([[float(self.levels[int(action)])]])
        if terminated or truncated:
            raise RuntimeError("The CityLearn episode ended before the horizon")
        cost = float(self.source.buildings[0].net_electricity_consumption_cost[self.steps])
        self.steps += 1
        obs, info = self._observe(source_obs)
        return obs, -cost, False, self.steps >= self.horizon, info

    def transition_metrics(self, info: dict) -> dict[str, float]:
        return {"net_consumption": float(info["net_consumption"])}

    def episode_metrics(self, *, total_reward, final_info, transition_metrics):
        del total_reward, final_info, transition_metrics
        return {"success": False}

    def make_policy(self, name: str, *, seed: int = 0, options: dict[str, Any] | None = None) -> Policy:
        if name == "random":
            return UniformRandom(seed=seed)
        if name == "time_of_use":
            options = dict(options or {})
            return TimeOfUse(
                levels=self.levels,
                charge_hours=options["tou_charge_hours"],
                charge_level=float(options["tou_charge_level"]),
                discharge_hours=options["tou_discharge_hours"],
                discharge_level=float(options["tou_discharge_level"]),
                seed=seed,
            )
        return super().make_policy(name, seed=seed, options=options)


class UniformRandom(Policy):
    """Uniform over the battery levels."""

    def action_probabilities(self, obs, info, deterministic=True):
        probabilities = masked_uniform(info["action_mask"])
        return greedy(probabilities) if deterministic else probabilities


class TimeOfUse(Policy):
    """A fixed schedule by hour of the day: charge while the sun is up, discharge in the price peak.

    The schedule of a home battery set to a time-of-use tariff. It reads only
    the hour, so it charges from the grid when the home uses its own solar
    power and discharges into the peak whether or not the home needs it.
    ``charge_hours`` and ``discharge_hours`` are hours ending (1-24), as the
    source's controllers use them; ``charge_level`` and ``discharge_level``
    must be battery levels of the action set.
    """

    def __init__(
        self,
        *,
        levels: np.ndarray,
        charge_hours: list[int],
        charge_level: float,
        discharge_hours: list[int],
        discharge_level: float,
        seed: int = 0,
    ):
        super().__init__(seed=seed)
        if set(charge_hours) & set(discharge_hours):
            raise ValueError("time_of_use charge and discharge hours overlap")
        self.num_actions = len(levels)
        self.idle = self._level_index(levels, 0.0)
        self.actions = {
            **{int(hour): self._level_index(levels, charge_level) for hour in charge_hours},
            **{int(hour): self._level_index(levels, discharge_level) for hour in discharge_hours},
        }

    @staticmethod
    def _level_index(levels: np.ndarray, level: float) -> int:
        matches = np.flatnonzero(np.isclose(levels, level))
        if len(matches) != 1:
            raise ValueError(f"time_of_use level {level} is not one of the battery levels {levels.tolist()}")
        return int(matches[0])

    def action_probabilities(self, obs, info, deterministic=True):
        hour_ending = int(round(float(obs[FEATURE_INDEX["hour"]]))) + 1
        return one_hot(self.num_actions, self.actions.get(hour_ending, self.idle))
