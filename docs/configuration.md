# Configuration

All configuration lives in `configs/` as small JSON files:

```text
configs/
  protocol.json              offline protocol shared by every run (includes `representation`)
  algorithms/<algo>.json     default hyperparameters of one algorithm
  envs/<env_id>.json         identity, run settings, feature_mix, normalization
  datasets/<env_id>.json     behavior policy x size tiers of one environment
  experiments/<name>.json    datasets x methods x representations x seeds, the
                             reference representation the others are compared to, and
                             optionally `train_episode_fraction` (low-data control)
```

## How a run config is resolved

`projeto-rl train <dataset> <method>` builds the run config in this order, each
layer overriding the previous one:

1. `protocol.json`
2. `algorithms/<algo>.json`
3. the `run` block of `envs/<env_id>.json` (`env_kwargs`, `environment_seed`, `gamma`)
4. identity and paths (`algo`, `method_id`, `dataset_id`, `seed`, dataset files)
5. `--set KEY=JSON` options from the command line (dotted keys reach nested objects)

In any of the files in steps 1-3, a `tiers` object holds values that apply
only to one tier (`dev` or `final`):

```json
{"policy_hidden_layers": 2, "tiers": {"dev": {"policy_hidden_dim": 128}, "final": {"policy_hidden_dim": 256}}}
```

Inspect the result without running anything:

```bash
projeto-rl train epicare/soc/dev dqn --dry-run --json
projeto-rl train simbank/bank/dev iql --set representation='"standardize"' --set offline_train_steps=500 --dry-run
```

The resolved config is saved as `config.json` in the run directory, and its
hash names the directory, so changing any value creates a new run.

## Environments

`envs/<env_id>.json` holds the `run` block, `feature_mix` (`homogeneous` or
`heterogeneous`, used to group the summary), and `normalization`: the
policies whose returns define scores 0 (`low`) and 100 (`high`). `high` must
be the dataset's behavior policy.

## Datasets

`datasets/<env_id>.json` declares the collection seed, the episode counts per
tier, and one behavior with its policy options: `behavior_policy`,
`behavior_epsilon` (uniform exploration mixed into it), and the policy's own
parameters (`mediocre_target`, `ticket_band_thresholds`, ...). The catalog is
every behavior times every tier.

## Experiments

See [experiments.md](experiments.md). `projeto-rl run-all <file>` runs one end to
end; `projeto-rl research matrix <file> --execute` runs only its training.

`projeto-rl validate` checks the files for consistency: unregistered environments,
missing algorithm defaults, undeclared behavior policies, a missing or invalid
`behavior_epsilon`, `feature_mix`, or `normalization`, and an unregistered
`representation` in the protocol.
