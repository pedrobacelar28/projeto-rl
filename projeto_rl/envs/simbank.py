"""SimBank: a loan-application process simulator with a true counterfactual outcome.

De Moor et al., "SimBank: from Simulation to Solution in Prescriptive Process
Monitoring" (arXiv:2506.14772); simulator vendored from the Zenodo record
10.5281/zenodo.15574272, v1.0.0, CC BY 4.0 (see third_party/simbank/NOTICE).

One episode is one loan application, driven through the source's online mode
(``start_simulation_inference`` -> ``continue_simulation_inference(action)``
-> ``end_simulation_inference``) with the ``time_contact_HQ`` intervention:
at each decision point the bank either does nothing (0) or contacts
headquarters (1). The observation is the last logged event before the
decision point: the current activity (categorical), the requested amount, the
estimated quality and its uncertainty (ordinal), the accumulated cost, the
elapsed days, the number of customer contacts, and the index of the decision
point. The reward is the case outcome (net present value of the loan, or minus
its costs when it is cancelled), divided by ``outcome_unit``, at the last step,
and zero before.

Adaptations and deviations:

- ``start_simulation_inference`` adds ``seed_to_add`` to the generator's
  stored seed, so seeds accumulate across calls; the adapter restores the base
  seed before every reset, so ``reset(seed)`` always yields the same case.
- Collection and evaluation use the guide's two base seeds (train 82, test
  130), selected by ``purpose``.
- About 7% of cases reach no decision point (the outcome does not depend on
  the policy); ``reset`` skips them deterministically.
- ``interest_rate``, ``discount_factor``, ``min_interest_rate`` (undefined
  before the offer) and ``nor`` (always 0 before the offer) are not observed;
  the true ``quality`` is hidden, as in the source.
- The ``random`` policy is the guide's random policy (one random decision
  point among the first ``action_depth``, then one random action that is kept
  for the rest of the case), not a uniform action per decision point: the
  bank policy does not beat the per-decision uniform policy (3361 +- 151 vs
  3701 +- 180 over 2000 cases) but does beat the source's random policy
  (2618 +- 150).
"""

from __future__ import annotations

import math
from datetime import datetime
from itertools import product
from typing import Any

import numpy as np
from simbank.simulation import PresProcessGenerator

from projeto_rl.envs import register_env
from projeto_rl.envs.base import EnvAdapter, EnvSpec, FeatureSpec
from projeto_rl.policies import Policy, greedy, one_hot

# Labelled transitions of the SimBank Petri net (petri_net_generator.create_PN), sorted.
ACTIVITIES = (
    "calculate_offer",
    "call_customer",
    "cancel_application",
    "contact_headquarters",
    "email_customer",
    "initiate_application",
    "receive_acceptance",
    "receive_refusal",
    "skip_contact",
    "start_priority",
    "start_standard",
    "validate_application",
)
FEATURES = (
    FeatureSpec("activity", "categorical", len(ACTIVITIES)),
    FeatureSpec("amount", "numeric"),
    FeatureSpec("est_quality", "ordinal", 11),
    FeatureSpec("unc_quality", "ordinal", 6),
    FeatureSpec("cum_cost", "numeric"),
    FeatureSpec("elapsed_time", "numeric"),
    FeatureSpec("noc", "numeric"),
    FeatureSpec("decision_index", "numeric"),
)
FEATURE_INDEX = {feature.name: index for index, feature in enumerate(FEATURES)}
# Offsets between retried reset seeds, far beyond any episode index in use.
SKIP_SEED_STRIDE = 10_000_019


def _dataset_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The guide notebook's ``dataset_params`` for one intervention, as the simulator expects them."""

    intervention = dict(kwargs["intervention"])
    info = {key: [value] for key, value in intervention.items()}
    info["retain_method"] = "precise"
    info["RCT"] = False
    info["action_combinations"] = list(product(*info["actions"]))
    info["action_width_combinations"] = math.prod(info["action_width"])
    info["action_depth_combinations"] = math.prod(info["action_depth"])
    info["len"] = [width if width > 2 else 1 for width in info["action_width"]]
    return {
        "log_cols": list(kwargs["log_cols"]),
        "simulation_start": datetime.fromisoformat(kwargs["simulation_start"]),
        "intervention_info": info,
        "policies_info": dict(kwargs["policies_info"]),
    }


@register_env
class SimBank(EnvAdapter):
    env_id = "simbank"
    behavior_policy_ids = {"random": 30, "bank": 31}

    def __init__(self, *, base_seed: int, seed: int | None = None, **kwargs):
        self.kwargs = kwargs
        self.params = _dataset_params(kwargs)
        self.intervention = dict(kwargs["intervention"])
        if self.intervention["name"] != "time_contact_HQ":
            raise ValueError("The SimBank adapter implements the time_contact_HQ intervention only")
        self.outcome_unit = float(kwargs["outcome_unit"])
        self.max_skipped_cases = int(kwargs["max_skipped_cases"])
        self.base_seed = int(base_seed)
        self.generator = PresProcessGenerator(self.params, self.base_seed)
        labels = sorted(t.label for t in self.generator.net.transitions if t.label and "ghost" not in t.label)
        if tuple(labels) != ACTIVITIES:
            raise RuntimeError(f"SimBank Petri net activities changed: {labels}")
        actions = self.intervention["actions"]
        self.spec = EnvSpec(
            env_id=self.env_id,
            num_actions=len(actions),
            features=FEATURES,
            action_labels=tuple(actions),
            metric_names=("cum_cost",),
            primary_metric="cum_cost",
        )
        self.reset(seed=0 if seed is None else seed)

    @classmethod
    def from_config(cls, config: dict[str, Any], *, seed: int | None = None, purpose: str) -> "SimBank":
        kwargs = dict(config["env_kwargs"])
        base_seeds = kwargs.pop("base_seeds")
        return cls(base_seed=int(base_seeds[purpose]), seed=seed, **kwargs)

    def _observe(self, event: dict[str, Any]) -> tuple[np.ndarray, dict]:
        values = {
            "activity": ACTIVITIES.index(event["activity"]),
            **{
                name: event[name]
                for name in ("amount", "est_quality", "unc_quality", "cum_cost", "elapsed_time", "noc")
            },
            "decision_index": self.decisions,
        }
        obs = np.asarray([float(values[feature.name]) for feature in FEATURES], dtype=np.float32)
        info = {
            "state_id": int(self.decisions),
            "action_mask": np.ones(self.num_actions, dtype=np.float32),
            "admissible_actions": list(range(self.num_actions)),
            "cum_cost": float(event["cum_cost"]),
        }
        return obs, info

    @staticmethod
    def _decision_event(prefix_list) -> dict[str, Any]:
        # Every branch shares the trace; its last event is the candidate
        # intervention, so the state is the event before it (as get_bank_best_action).
        return prefix_list[0][0:-1][-1]

    def reset(self, seed: int | None = None):
        seed = 0 if seed is None else int(seed)
        for attempt in range(self.max_skipped_cases + 1):
            self.generator.random_seed = self.base_seed
            prefix = self.generator.start_simulation_inference(seed_to_add=seed + attempt * SKIP_SEED_STRIDE)
            if self.generator.int_points_available:
                break
        else:
            raise RuntimeError(f"SimBank produced {self.max_skipped_cases + 1} cases without a decision point")
        self.decisions = 0
        return self._observe(self._decision_event(prefix))

    def step(self, action: int):
        prefix = self.generator.continue_simulation_inference(int(action))
        self.decisions += 1
        if self.generator.int_points_available:
            obs, info = self._observe(self._decision_event(prefix))
            return obs, 0.0, False, False, info
        case = self.generator.end_simulation_inference()
        outcome = float(case[-1]["outcome"])
        obs, info = self._observe(case[-1])
        info["outcome"] = outcome
        return obs, outcome / self.outcome_unit, True, False, info

    def transition_metrics(self, info: dict) -> dict[str, float]:
        return {"cum_cost": float(info["cum_cost"])}

    def episode_metrics(self, *, total_reward, final_info, transition_metrics):
        del transition_metrics
        return {"success": bool(total_reward > 0.0), "outcome": float(final_info["outcome"])}

    def make_policy(self, name: str, *, seed: int = 0, options: dict[str, Any] | None = None) -> Policy:
        if name == "bank":
            return BankPolicy(self.params["policies_info"], seed=seed)
        if name == "random":
            return GuideRandom(depth=int(self.intervention["action_depth"]), width=self.num_actions, seed=seed)
        return super().make_policy(name, seed=seed, options=options)


class BankPolicy(Policy):
    """The bank's rule for ``time_contact_HQ`` (``get_bank_best_action`` in the SimBank guide notebook).

    Cancel-like cases are never escalated; small-uncertainty, sizeable, good
    cases with fewer than two customer contacts are sent to headquarters.
    """

    def __init__(self, policies_info: dict[str, Any], *, seed: int = 0):
        super().__init__(seed=seed)
        self.min_quality = float(policies_info["min_quality"])
        self.max_noc = float(policies_info["max_noc"])

    def action_probabilities(self, obs, info, deterministic=True):
        amount, est_quality, unc_quality, noc = (
            float(obs[FEATURE_INDEX[name]]) for name in ("amount", "est_quality", "unc_quality", "noc")
        )
        cancel = (unc_quality == 0 and est_quality < self.min_quality and noc >= self.max_noc) or (
            noc >= self.max_noc and unc_quality > 0
        )
        contact = noc < 2 and unc_quality == 0 and amount > 10000 and est_quality >= self.min_quality
        return one_hot(len(info["action_mask"]), 1 if (contact and not cancel) else 0)


class GuideRandom(Policy):
    """The SimBank guide's random policy for ``time_contact_HQ``.

    Per case it draws one decision point ``2 * k``, ``k`` uniform in
    ``0..depth-1``; before it the action is 0, at it the action is uniform, and
    the drawn action is repeated afterwards. The guide's factor 2 is kept.
    """

    def __init__(self, *, depth: int, width: int, seed: int = 0):
        super().__init__(seed=seed)
        self.depth = int(depth)
        self.width = int(width)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.timing = 2 * int(self.rng.integers(self.depth))
        self.position = 0
        self.chosen = 0

    def observe(self, action: int, reward: float) -> None:
        if self.position == self.timing:
            self.chosen = int(action)
        self.position += 1

    def action_probabilities(self, obs, info, deterministic=True):
        if self.position == self.timing:
            probabilities = np.full(self.width, 1.0 / self.width)
            return greedy(probabilities) if deterministic else probabilities
        return one_hot(self.width, self.chosen)
