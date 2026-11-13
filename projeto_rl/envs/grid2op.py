"""Grid2Op: keeping a 14-substation power grid in service by switching substation topologies.

Donnot, "Grid2Op: A testbed platform to model sequential decision making in
power systems", 2020 (github.com/rte-france/Grid2Op), MPL-2.0; Marot et al.,
"Learning to run a power network challenge for training topology
controllers", arXiv:1912.04211. Simulator: the ``grid2op`` package 1.12.6
with the ``lightsim2grid`` backend, installed, not vendored; environment
``l2rpn_case14_sandbox`` (the IEEE 14-bus grid with 6 generators and 11
loads), downloaded by the package. Its 1004 scenarios of 28 days at 5-minute
steps are synthetic series; the download states no license for them.

One episode is one day (``horizon`` steps) of one scenario from a start drawn
in the scenario. The actions are doing nothing and the ``actions`` of the
config, each setting the buses of one substation (a unitary topology of
``get_all_unitary_topologies_set``). The action mask removes what the source's
``DefaultRules`` would refuse: an action that touches a line still in its
reconnection cooldown, or more than one line or substation. The observation is
153 columns of the source's ``CompleteObservation`` (``FEATURES``): the time
of day and day of the week, generator and load injections, load voltages and
angles, line flows and loadings (``rho``), the bus of every element that the
actions move (1 or 2, or 0 when disconnected), and the status and reconnection
cooldown of line 13, the only line that the behavior data disconnects. The reward is the
source's ``L2RPNReward`` divided by the number of lines: the mean over lines of
``max(1 - min(|a| / (limit + 0.1), 1)^2, 0)``, from 1 for idle lines down to 0
for lines at their thermal limit. A blackout (the source's game over) ends the
episode as a terminal state with reward 0.

Adaptations and deviations:

- The source's reward is 0 at the last step of an episode even when the grid
  survives; the reward here is computed with the source's formula from the
  observation at every step, so the last step counts like the others.
- The 1004 scenarios are split once: ``collect`` scenarios for datasets and
  ``evaluate`` scenarios for simulator scores.
- The columns that never vary in the source's observation (the year, the
  maintenance and redispatch fields, alarms, storage), the ones that copy
  another column (``a_or``, ``a_ex``, ``p_ex``, ``q_ex``, ``theta_or``,
  ``theta_ex``), and the elements that stay constant (the bus of elements no
  action moves, the voltage of load 1, the angle of generators 1 and 5) are
  left out. ``timestep_overflow`` is left out: it is non-zero in fewer than
  0.3% of the rows.
- The 40 actions are the ones a greedy expert (the ``rule`` policy below over
  all 178 unitary topologies, threshold 0.9) chose most often on the collect
  scenarios 0-799: they cover 90% of its actions.
- The behavior policy reads the simulator's one-step look-ahead
  (``obs.simulate``), as L2RPN baselines do; it only logs data, and the
  learned agents never call it.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

with warnings.catch_warnings():
    # pandapower, imported by grid2op, warns that the optional numba is missing.
    warnings.filterwarnings("ignore", message="Numba cannot be loaded")
    import grid2op
from lightsim2grid import LightSimBackend

from projeto_rl.envs import register_env
from projeto_rl.envs.base import EnvAdapter, EnvSpec, FeatureSpec
from projeto_rl.policies import Policy, greedy, masked_uniform, one_hot

# (observation attribute, elements, kind, cardinality); the source's element order.
COLUMNS = (
    ("hour_of_day", None, "cyclic", 24),
    ("minute_of_hour", None, "cyclic", 12),
    ("day_of_week", None, "cyclic", 7),
    ("gen_p", range(6), "numeric", None),
    ("gen_q", range(6), "numeric", None),
    ("load_p", range(11), "numeric", None),
    ("load_q", range(11), "numeric", None),
    ("load_v", (0, *range(2, 11)), "numeric", None),
    ("p_or", range(20), "numeric", None),
    ("q_or", range(20), "numeric", None),
    ("rho", range(20), "numeric", None),
    ("load_theta", range(11), "numeric", None),
    ("gen_theta", (0, 2, 3, 4), "numeric", None),
    (
        "topo_vect",
        (*range(4, 9), *range(14, 19), *range(20, 24), *range(25, 31), *range(37, 41), 42, 43, 48, 51, 52),
        "categorical",
        3,
    ),
    ("line_status", (13,), "binary", None),
    ("time_before_cooldown_line", (13,), "ordinal", 11),
)
FEATURES = tuple(
    FeatureSpec(attribute if elements is None else f"{attribute}_{element}", kind, cardinality)
    for attribute, elements, kind, cardinality in COLUMNS
    for element in (elements or (None,))
)
MINUTES_PER_STEP = 5


def observation_vector(obs) -> np.ndarray:
    """The 153 columns of one grid2op observation, coded as ``FEATURES`` declares."""

    values = []
    for attribute, elements, kind, _ in COLUMNS:
        raw = np.asarray(getattr(obs, attribute), dtype=np.float64)
        column = raw[None] if elements is None else raw[list(elements)]
        if attribute == "minute_of_hour":
            column = column / MINUTES_PER_STEP
        elif attribute == "topo_vect":
            # Buses are 1 or 2, and -1 for a disconnected element.
            column = np.where(column < 0, 0.0, column)
        values.append(column)
    return np.concatenate(values).astype(np.float32)


def line_scores(obs, thermal_limits: np.ndarray) -> np.ndarray:
    """``L2RPNReward``'s per-line score, from the observed currents."""

    relative_flow = np.abs(np.asarray(obs.a_or, dtype=np.float64)) / (thermal_limits + 0.1)
    return np.maximum(1.0 - np.minimum(relative_flow, 1.0) ** 2, 0.0)


@register_env
class Grid2Op(EnvAdapter):
    env_id = "grid2op"
    behavior_policy_ids = {"random": 60, "rule": 61}

    def __init__(
        self,
        *,
        dataset: str,
        scenarios: list[int],
        horizon: int,
        actions: list[list],
        seed: int | None = None,
    ):
        low, high = (int(value) for value in scenarios)
        if int(horizon) <= 0 or not actions:
            raise ValueError("grid2op needs horizon > 0 and at least one action")
        self.source = grid2op.make(str(dataset), backend=LightSimBackend())
        self.horizon = int(horizon)
        self.scenario_length = int(self.source.chronics_handler.max_timestep())
        num_scenarios = len(self.source.chronics_handler.real_data.subpaths)
        if not 0 <= low < high <= num_scenarios or self.horizon > self.scenario_length:
            raise ValueError(f"Scenarios {scenarios} or horizon {horizon} do not fit {num_scenarios} scenarios")
        self.scenarios = (low, high)
        parameters = self.source.parameters
        if max(parameters.NB_TIMESTEP_COOLDOWN_LINE, parameters.NB_TIMESTEP_RECONNECTION) + 1 != 11:
            raise RuntimeError("The line cooldown of the grid2op parameters changed; update FEATURES")
        self.max_lines_changed = int(parameters.MAX_LINE_STATUS_CHANGED)
        self.max_substations_changed = int(parameters.MAX_SUB_CHANGED)
        self.thermal_limits = np.asarray(self.source.get_thermal_limit(), dtype=np.float64)
        space = self.source.action_space
        self.actions = [space({})]
        for substation, buses in actions:
            if len(buses) != int(space.sub_info[int(substation)]):
                raise ValueError(f"Action on substation {substation} needs {space.sub_info[int(substation)]} buses")
            self.actions.append(space({"set_bus": {"substations_id": [(int(substation), [int(b) for b in buses])]}}))
        self.spec = EnvSpec(
            env_id=self.env_id,
            num_actions=len(self.actions),
            features=FEATURES,
            action_labels=(
                "do nothing",
                *(f"substation {substation} buses {''.join(map(str, buses))}" for substation, buses in actions),
            ),
            metric_names=("max_rho",),
            primary_metric="max_rho",
        )
        self.reset(seed=seed)

    @classmethod
    def from_config(cls, config: dict[str, Any], *, seed: int | None = None, purpose: str) -> "Grid2Op":
        kwargs = dict(config["env_kwargs"])
        scenarios = kwargs.pop("scenarios")[purpose]
        return cls(scenarios=scenarios, seed=seed, **kwargs)

    def close(self) -> None:
        self.source.close()

    def legal_mask(self, obs) -> np.ndarray:
        """Actions ``DefaultRules`` accepts in ``obs``: at most one line and substation, none in cooldown."""

        mask = np.ones(self.num_actions, dtype=np.float32)
        for index, action in enumerate(self.actions[1:], start=1):
            lines, substations = action.get_topological_impact(
                obs.line_status, _store_in_cache=False, _read_from_cache=False
            )
            if (
                lines.sum() > self.max_lines_changed
                or substations.sum() > self.max_substations_changed
                or (obs.time_before_cooldown_line[lines] > 0).any()
                or (obs.time_before_cooldown_sub[substations] > 0).any()
            ):
                mask[index] = 0.0
        return mask

    def _observe(self, obs, *, blackout: bool) -> tuple[np.ndarray, dict]:
        # After a blackout no action is admissible; the mask keeps doing
        # nothing so that it never sums to zero.
        mask = np.eye(self.num_actions, dtype=np.float32)[0] if blackout else self.legal_mask(obs)
        info = {
            "state_id": int(self.scenario),
            "action_mask": mask,
            "admissible_actions": np.flatnonzero(mask).tolist(),
            "max_rho": float(np.max(obs.rho)),
            "grid_observation": obs,
            "grid_actions": self.actions,
        }
        return observation_vector(obs), info

    def reset(self, seed: int | None = None):
        rng = np.random.default_rng(seed)
        low, high = self.scenarios
        self.scenario = int(rng.integers(low, high))
        self.start = int(rng.integers(self.scenario_length - self.horizon + 1))
        options = {"time serie id": self.scenario, "max step": self.horizon}
        if self.start > 0:
            # The source refuses "init ts" = 0.
            options["init ts"] = self.start
        obs = self.source.reset(seed=int(rng.integers(2**31)), options=options)
        self.steps = 0
        self.blackout = False
        return self._observe(obs, blackout=False)

    def step(self, action: int):
        obs, _, done, source_info = self.source.step(self.actions[int(action)])
        if source_info["is_illegal"] or source_info["is_ambiguous"]:
            raise RuntimeError(f"grid2op refused action {action} that the mask allowed: {source_info['exception']}")
        self.steps += 1
        self.blackout = bool(done and source_info["exception"])
        if done and not self.blackout and self.steps < self.horizon:
            raise RuntimeError(f"The grid2op episode ended at step {self.steps} without a blackout")
        reward = 0.0 if self.blackout else float(line_scores(obs, self.thermal_limits).mean())
        obs_vector, info = self._observe(obs, blackout=self.blackout)
        return obs_vector, reward, self.blackout, not self.blackout and self.steps >= self.horizon, info

    def transition_metrics(self, info: dict) -> dict[str, float]:
        return {"max_rho": float(info["max_rho"])}

    def episode_metrics(self, *, total_reward, final_info, transition_metrics):
        del total_reward, final_info, transition_metrics
        return {"success": not self.blackout}

    def make_policy(self, name: str, *, seed: int = 0, options: dict[str, Any] | None = None) -> Policy:
        if name == "random":
            return UniformRandom(seed=seed)
        if name == "rule":
            return Rule(threshold=float(dict(options or {})["rule_threshold"]), seed=seed)
        return super().make_policy(name, seed=seed, options=options)


class UniformRandom(Policy):
    """Uniform over the admissible actions."""

    def action_probabilities(self, obs, info, deterministic=True):
        probabilities = masked_uniform(info["action_mask"])
        return greedy(probabilities) if deterministic else probabilities


class Rule(Policy):
    """Do nothing while every line is below ``threshold`` of its limit; otherwise take the best simulated action.

    Above the threshold the rule simulates one step of doing nothing and of
    every admissible action with the source's forecasts (``obs.simulate``) and
    takes the action with the lowest simulated maximum line loading, if it is
    lower than doing nothing's. A simulation that fails counts as an infinite
    loading. Over all 178 unitary topologies, this is the expert that chose
    the action set.
    """

    def __init__(self, *, threshold: float, seed: int = 0):
        super().__init__(seed=seed)
        self.threshold = float(threshold)

    def action_probabilities(self, obs, info, deterministic=True):
        mask = info["action_mask"]
        if info["max_rho"] <= self.threshold:
            return one_hot(len(mask), 0)
        grid_obs, actions = info["grid_observation"], info["grid_actions"]
        loadings = np.full(len(mask), np.inf)
        for index in np.flatnonzero(mask):
            simulated, _, done, simulate_info = grid_obs.simulate(actions[index])
            if not (done or simulate_info["exception"]):
                loadings[index] = float(np.max(simulated.rho))
        best = int(np.argmin(loadings))
        return one_hot(len(mask), best if loadings[best] < loadings[0] else 0)
