# Tabular state representations for offline RL

**Question.** Does a better vector representation of tabular (structured)
states improve the return of policies learned offline?

**Protocol.** Each of six simulators provides one fixed offline dataset,
logged by a behavior policy. Each classic offline RL method (`bc`, `dqn`,
`iql`, `cql`) is trained on that dataset twice with the same seeds and
hyperparameters: once on the standardized observation (`representation =
"standardize"`, the reference) and once on the observation transformed by
each representation under study. The checkpoint is
selected by the offline validation loss, and the deployed policy is scored by
its mean return in the online simulator, also reported as a normalized score:

```
score = 100 * (R - R_random) / (R_behavior - R_random)
```

0 means "as good as the random policy", 100 "as good as the policy that logged
the data". Each representation is compared with `standardize` by a paired
difference over seeds. There is no off-policy evaluation: the simulator is the judge, and
it is never used for training or model selection.

The repository ships six representations: `raw` (identity), `standardize`
(z-score fitted on the train split), `spr` (a latent encoder pretrained on
the logged transitions with the self-predictive objective of Schwarzer et al.,
2021), `tjepa` (a transformer over column tokens pretrained to predict
the latent of masked columns, Thimonier et al., 2025), and `tabpfn` (row
embeddings of the pretrained TabPFN-3.5 model given logged train rows; its
weights are non-commercial and need a one-time license acceptance, without
which the run warns and uses TabPFN-2 instead, see
`docs/representations.md`), and `tdjepa` (a state encoder pretrained on the
logged transitions to predict, by temporal differences, the latents that a
family of policies will visit, Bagatella et al., 2026). A first experiment showed that `raw`
breaks the networks on environments whose columns have very different scales,
so `standardize` is the reference.
Tabular representation techniques are added as new modules (see
[docs/representations.md](docs/representations.md)).

It is research code for a university project, not a decision-support system.

## Environments

Generated from `configs/` and the environment registry (`projeto-rl catalog`
lists the datasets):

| Environment | Columns (type) | Actions | Group | Behavior policy (dataset) | Score 0 / 100 |
|---|---|---:|---|---|---|
| `epicare` | 8: 8 numeric | 16 | homogeneous | `soc` + ε=0.1 | `random` / `soc` |
| `industrial-benchmark` | 6: 6 numeric | 27 | homogeneous | `mediocre` + ε=0.1 | `random` / `mediocre` |
| `sales-promotion` | 5: 4 numeric, 1 cyclic | 48 | heterogeneous | `ticket_band` + ε=0.1 | `random` / `ticket_band` |
| `simbank` | 8: 1 categorical, 5 numeric, 2 ordinal | 2 | heterogeneous | `bank` + ε=0.1 | `random` / `bank` |
| `citylearn` | 28: 3 cyclic, 25 numeric | 11 | heterogeneous | `time_of_use` + ε=0.1 | `random` / `time_of_use` |
| `grid2op` | 153: 1 binary, 29 categorical, 3 cyclic, 119 numeric, 1 ordinal | 41 | heterogeneous | `rule` + ε=0.1 | `random` / `rule` |

The two homogeneous tables (only numeric columns) are the contrast for the
heterogeneous ones (categorical, ordinal, cyclic, and heavy-tailed numeric
columns). `citylearn` and `grid2op` are the wide tables: they test whether the
representations need more columns than the 5 to 8 of the first four. Sources, column meanings, and every adaptation are in
[docs/environments.md](docs/environments.md).

| Method | Summary |
|---|---|
| `bc` | Behavior cloning: masked cross-entropy on logged actions |
| `dqn` | Offline DQN: masked one-step TD regression with a soft-updated target network |
| `iql` | Discrete IQL: expectile value, twin critics, advantage-weighted actor |
| `cql` | Discrete CQL: TD regression plus a conservative logsumexp penalty |

## Install

```bash
conda create -n projeto-rl python=3.13 && conda activate projeto-rl
pip install -r requirements-lock.txt                # pinned reference versions
pip install --no-deps -r requirements-nodeps.txt    # CityLearn and Grid2Op (their pins predate numpy 2 and pandas 3)
pip install -e . --no-deps                          # projeto_rl + the four vendored simulators
```

This installs the `projeto-rl` command (also `python -m projeto_rl`).

## Run the experiment

```bash
projeto-rl run-all configs/experiments/final.json --dry-run     # what is left to do
projeto-rl run-all configs/experiments/final.json --gpus 0 --jobs 32
```

`run-all` builds the six datasets, evaluates the random and behavior policies
that normalize the scores, trains 6 environments x 4 methods x 6
representations x 3 seeds = 432 runs, and prints the summary tables. The
terminal shows the current stage, the elapsed time of the stage and of the
whole run, and the estimated time left for training. Interrupting it and
running the same command again resumes where it stopped. For a long run:

```bash
nohup projeto-rl run-all configs/experiments/final.json --gpus 0 --jobs 32 > artifacts/run-all.log 2>&1 &
tail -f artifacts/run-all.log
```

`configs/experiments/dev.json` is the same experiment on small datasets with
one seed (it is the smoke test). `configs/experiments/wide.json` is the
`final` protocol on the two wide environments only (144 runs), and
`configs/experiments/wide-dev.json` its smoke test.

Results land in `artifacts/reports/summary/<timestamp>/`: `summary.md` (the
tables), `runs.csv`, `by_method.csv`, `comparison.csv`, `summary.json`, and
`loss_curves/<dataset>.png` (training and validation loss of every run).

## Step by step

```bash
projeto-rl dataset build simbank/bank/dev
projeto-rl baselines simbank/bank/dev
projeto-rl train simbank/bank/dev dqn --set representation='"standardize"'
projeto-rl research summarize configs/experiments/dev.json
```

## Repository layout

```text
projeto_rl/
  envs/             environment registry and one adapter per simulator
  representations/  representation registry: raw, standardize, ...
  algorithms/       bc, dqn, iql, cql
  data/             dataset schema, loading, and collection
  training.py       the offline training loop (fits the representation on train)
  evaluation.py     simulator evaluation
  runner.py         one run: train -> evaluate -> artifacts
  config.py         layered configs, the derived catalog, experiment files
  operations/       run artifacts, job queue, baselines, summary statistics
  cli/              the `projeto-rl` command
configs/            protocol, algorithms, environments, datasets, experiments
third_party/        vendored simulators (see THIRD_PARTY_NOTICES.md)
tests/              contract and known-truth tests
docs/               documentation
artifacts/          generated datasets, runs, reports (git-ignored)
```

## Documentation

- [Environments](docs/environments.md)
- [Representations](docs/representations.md): the contract and how to add one
- [Experiments](docs/experiments.md): protocol, normalization, statistics
- [Algorithms](docs/algorithms.md)
- [Configuration](docs/configuration.md)
- [Data schema](docs/data-schema.md)
- [Command line and artifacts](docs/operations.md)
- [Development and tests](docs/development.md)

## Third-party code

- **Vendored simulators** keep their licenses (Apache-2.0, MIT, CC BY 4.0);
  see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **`pm4py`**, needed by SimBank, is installed as a dependency and is
  AGPL-3.0-or-later.
- **CityLearn** (MIT) and **Grid2Op** (MPL-2.0, with `lightsim2grid`,
  MPL-2.0, and `pandapower`, BSD-3-Clause) are installed, not vendored; they
  download their datasets on first use.
