# Environments

Four simulators with a true simulator from the source, so every policy is
scored online. Each is one adapter in `projeto_rl/envs/<name>.py` (a subclass of
`EnvAdapter`, see `projeto_rl/envs/base.py`), a run config in
`configs/envs/<env_id>.json`, and a dataset config in
`configs/datasets/<env_id>.json`. The vendored code and every change made to
it are listed in `third_party/<name>/NOTICE`.

Every observation column has a declared type (`FeatureSpec`: `numeric`,
`binary`, `ordinal`, `categorical`, or `cyclic`, with a cardinality for the
last three). The types are stored in every dataset's `metadata.json` and are
what representations read. The contract tests check every rollout against
them.

## `epicare` (homogeneous)

EpiCare (Hargrave, Spaeth, and Grosenick, NeurIPS 2024 Datasets and
Benchmarks), Apache-2.0, vendored. A procedurally generated longitudinal
treatment POMDP: the agent sees 8 symptom intensities in [0, 1] (all
`numeric`), picks one of 16 treatments, and the hidden disease is only a
diagnostic `state_id`. Episodes last up to 8 visits.

- Dataset behavior: `soc`, the source's standard of care (adaptive and
  deterministic), mixed with 10% uniform exploration.
- Normalization: `random` (uniform over the 16 treatments) and `soc`.

## `industrial-benchmark` (homogeneous)

Industrial Benchmark (Hein et al., IEEE SSCI 2017, arXiv:1709.09480),
Python implementation from `github.com/siemens/industrialbenchmark`, MIT.

- Observation: the six current variables `setpoint`, `velocity`, `gain`,
  `shift`, `fatigue`, `consumption` (all `numeric`); no 30-step history.
- Actions: 27 = {-1, 0, +1} on velocity, gain, and shift, the source's
  `IBGym(action_type="discrete")` table.
- Reward: minus the cost (`reward_type="classic"`). Episodes have
  `horizon = 250` steps (truncated) and draw the setpoint from
  `setpoints`; `gamma = 0.99`.
- Dataset behavior: `mediocre`, an operator rule that moves each lever
  toward 50 with a dead band of 5 (the repository ships no behavior
  policies), mixed with 10% uniform exploration. The task has no success
  event (`success` is always false).

## `simbank` (heterogeneous)

SimBank (De Moor et al., arXiv:2506.14772), Zenodo
10.5281/zenodo.15574272 v1.0.0, CC BY 4.0. A loan-application process with a
true counterfactual outcome, run through the source's online mode with the
`time_contact_HQ` intervention: at each decision point the bank does nothing
(0) or contacts headquarters (1).

- Observation: the last event before the decision point: `activity`
  (categorical, 12 activities of the Petri net), `amount`,
  `est_quality` (ordinal 0-10), `unc_quality` (ordinal 0-5), `cum_cost`,
  `elapsed_time`, `noc` (customer contacts), and `decision_index`.
- Reward: the case outcome (net present value of an accepted loan, or minus
  the costs otherwise) in thousands, at the last step. Episodes have 1-8
  decision points; about 7% of cases have none and are skipped.
- Dataset behavior: `bank`, the bank rule of the SimBank guide notebook,
  mixed with 10% uniform exploration per decision point.
- Normalization: `random` is the guide's random policy (one random decision
  point, one random action that is kept afterwards). A uniform action per
  decision point is *not* used as the zero: it beats the bank rule (3701 ±
  180 vs 3361 ± 151 over 2000 cases), while the guide's random policy does
  not (2618 ± 150).
- Collection and evaluation use the guide's two base seeds (82 and 130).

## `sales-promotion` (heterogeneous)

NeoRL SalesPromotion (Qin et al., NeurIPS 2022 Datasets and Benchmarks),
`benchmark` branch, Apache-2.0: a coupon-allocation simulator driven by an
ensemble user model learned from real users.

- One episode is one user over the source's 30 evaluation days. The source
  steps 10 000 users together and returns the summed reward; the adapter
  returns each user's own reward.
- Observation: the source's `total_orders`, `avg_orders`, `avg_fee`
  (`numeric`) and `weekday` (`cyclic`, 7), plus the episode `day`. The user
  model keeps an 18-column hidden state, so the task is partially observable.
- Actions: 48 = 6 coupon counts (0-5) x 8 discounts (0.95 ... 0.60, the
  source's `DISCOUNT_COUPON_LIST`).
- Reward: the source's `gmv - cost` per user and day.
- Users 0-7999 are used for datasets and users 8000-9999 for evaluation
  (`purpose` in `make_env`), so scores are on unseen users.
- Dataset behavior: `ticket_band`, a fixed offer per tercile of `avg_fee`
  (5 coupons at 0.70, 3 at 0.85, 1 at 0.95), mixed with 10% uniform
  exploration. The source's human-operator logs are not used: their policy
  cannot be evaluated in the simulator, so it could not anchor the score.

## Adding an environment

Write `projeto_rl/envs/<name>.py` with an `EnvAdapter` subclass decorated with
`@register_env` (typed `features`, `from_config(config, *, seed, purpose)`,
`reset`, `step`, `make_policy`), import it at the bottom of
`projeto_rl/envs/__init__.py`, add `configs/envs/<env_id>.json` (with
`feature_mix` and `normalization`) and `configs/datasets/<env_id>.json`, and
add a tiny dataset to `tests/conftest.py`. The contract tests then cover it.
