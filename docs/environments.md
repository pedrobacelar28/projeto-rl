# Environments

Six simulators with a true simulator from the source, so every policy is
scored online. The first four have 5 to 8 columns; `citylearn` (28) and
`grid2op` (153) are the wide tables. Each is one adapter in `projeto_rl/envs/<name>.py` (a subclass of
`EnvAdapter`, see `projeto_rl/envs/base.py`), a run config in
`configs/envs/<env_id>.json`, and a dataset config in
`configs/datasets/<env_id>.json`. The vendored code and every change made to
it are listed in `third_party/<name>/NOTICE`; CityLearn and Grid2Op are
installed packages (`requirements-nodeps.txt`) that download their data on
first use (`~/.cache/citylearn`, about 3 MB; `~/data_grid2op`, about 300 MB).

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

## `citylearn` (heterogeneous, 28 columns)

CityLearn 3.2.0 (Vázquez-Canteli et al., BuildSys 2019; Nweye et al., PMLR
220, 2023), MIT, installed. Dataset `citylearn_challenge_2022_phase_all`: one
year (August 2016 to July 2017) of hourly load and solar generation measured
in 17 single-family homes in Fontana, California, cleaned by EPRI (gaps and
outliers filled, some values by a fitted model), with shared weather, carbon
intensity, and a time-of-use tariff. Each home has a 6.4 kWh, 5 kW battery
(simulated) and 4 or 5 kW of solar panels.

- One episode is one home over one week (`horizon = 168` hours) from an hour
  drawn in the year, with the battery empty; the end is a time limit
  (`timeouts`), `gamma = 0.99`.
- Observation: the source's 28 active columns: `month` (`cyclic`, 12),
  `day_type` (weekday, `cyclic`, 7), `hour` (`cyclic`, 24), outdoor
  temperature, humidity, diffuse and direct irradiance with their 6, 12, and
  24 h ahead values (the dataset's "predictions" are the actual future
  values), carbon intensity, the home's load, solar generation, battery state
  of charge, net consumption (the last two of the previous hour, as in the
  source), and the price with its three ahead values (all `numeric`).
- Actions: 11 battery levels, `linspace(-1, 1, 11)` times the 5 kW nominal
  power for one hour (positive charges); the battery model clips what it
  cannot do.
- Reward: minus the hour's bill, price times the net consumption, with
  exported energy earning nothing (the source's
  `net_electricity_consumption_cost`).
- Homes 1-12 are used for datasets and homes 13-17 for evaluation. Homes 14
  and 15 have much lower solar output per kW than the others (a property of
  the data).
- Dataset behavior: `time_of_use`, a home-battery schedule for the tariff
  (charge 1 kWh per hour ending 10-15, discharge 2 kWh per hour ending 16-20,
  idle otherwise), mixed with 10% uniform exploration. The source's
  `BasicRBC` and `OptimizedRBC` were written for an older action unit and do
  worse than an idle battery, so they are not used. Measured over 40 weeks per
  home with `time_of_use` as 100 and uniform random as 0: an idle battery
  scores 77-130 and a self-consumption rule (charge the solar surplus,
  discharge the deficit) 108-135, so there is room above the behavior.
- Normalization: `random` (uniform over the 11 levels) and `time_of_use`.

## `grid2op` (heterogeneous, 153 columns)

Grid2Op 1.12.6 (Donnot, 2020; Marot et al., arXiv:1912.04211) with the
`lightsim2grid` 1.3.0 backend, MPL-2.0, installed. Environment
`l2rpn_case14_sandbox`: the IEEE 14-bus grid with 20 lines, 6 generators,
and 11 loads, and 1004 scenarios of 28 days at 5-minute steps. The scenarios
are synthetic (one stationary winter-like generator, the same calendar in
every scenario), and the download states no license for them.

- One episode is one day (`horizon = 288` steps) of one scenario from a start
  drawn in it. A blackout (the source's game over) ends it as a terminal
  state with reward 0; surviving the day ends it as a time limit and counts as
  `success`. `gamma = 0.99`.
- Observation (153 columns of the source's `CompleteObservation`):
  `hour_of_day` (`cyclic`, 24), `minute_of_hour` (`cyclic`, 12 five-minute
  codes), `day_of_week` (`cyclic`, 7); generator and load active and reactive
  power, load voltages and angles, generator angles, line active and reactive
  flows at the origin, and line loading `rho` (`numeric`); the bus of the 29
  elements that the actions move (`categorical`, 3: disconnected, bus 1, bus
  2); the status (`binary`) and reconnection cooldown (`ordinal`, 0-10) of
  line 13, the only line that the behavior data disconnects. Constant
  columns and copies of other columns are left out (the adapter's docstring
  lists them).
- Actions: 41 = do nothing plus 40 substation topologies, the ones that a
  greedy expert (the `rule` below over all 178 unitary topologies) chose most
  often on scenarios 0-799; they cover 90% of its actions. The action mask
  removes what the source's rules refuse (touching a line in its reconnection
  cooldown, or more than one line or substation).
- Reward: the source's `L2RPNReward` divided by the 20 lines, the mean over
  lines of `max(1 - min(|a| / (limit + 0.1), 1)^2, 0)`, in [0, 1] per step.
  The source gives 0 at the last step of a day the grid survives; the adapter
  scores that step like the others.
- Scenarios 0-899 are used for datasets and 900-1003 for evaluation.
- Dataset behavior: `rule`: do nothing while every line is at most 90% loaded;
  otherwise simulate one step of every admissible action with the source's
  forecasts (`obs.simulate`) and take the one with the lowest maximum
  loading. Only the behavior reads the look-ahead; learned agents never see
  it. Mixed with 10% uniform exploration over the admissible actions, as in
  every environment. The rule alone never lost the grid in the measurements
  (30 one-day episodes), but a random topology change often trips it, so the
  logged days end in a blackout after about 90 steps on average: the dataset
  is mostly the first hours of days that fail. Uniform random loses the grid
  within about 5 steps; doing nothing survives 251 of 288 steps on average.
- Normalization: `random` (uniform over the admissible actions) and `rule`
  (the ε-mixed behavior, as everywhere).

## Adding an environment

Write `projeto_rl/envs/<name>.py` with an `EnvAdapter` subclass decorated with
`@register_env` (typed `features`, `from_config(config, *, seed, purpose)`,
`reset`, `step`, `make_policy`), import it at the bottom of
`projeto_rl/envs/__init__.py`, add `configs/envs/<env_id>.json` (with
`feature_mix` and `normalization`) and `configs/datasets/<env_id>.json`, and
add a tiny dataset to `tests/conftest.py`. The contract tests then cover it.
Every column must vary in the tiny train split, because `standardize` rejects
constant columns.
