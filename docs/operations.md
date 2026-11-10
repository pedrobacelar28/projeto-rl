# Command Line and Artifacts

`projeto-rl` (equivalently `python -m projeto_rl`). Global options, accepted before or
after the subcommand: `--artifact-root` (default `artifacts/`),
`--config-root` (default `configs/`), `--json`, and `--color never`.

## Commands

| Command | Purpose |
|---|---|
| `run-all <experiment.json>` | Whole experiment: datasets, baselines, training, summary (`--gpus`, `--jobs`, `--dry-run`) |
| `doctor` | Packages, CUDA/GPUs, registered environments and algorithms, config errors |
| `catalog` | Datasets (with build state) and methods |
| `validate [--smoke]` | Check configs; optionally step every environment once |
| `dataset list \| show <id> \| build <id>` | Discover and collect datasets |
| `inspect dataset <id>` | Array shapes and dtypes of a built dataset |
| `baselines <id>` | Evaluate the random and behavior policies that normalize the dataset's scores |
| `train <dataset> <method>` | One run: train, then evaluate in the simulator |
| `suite <dataset>` | Queue every (or `--method`-selected) method over `--seeds` |
| `status [run]` | Recent runs and their status |
| `inspect run <run>` | Optimization summary and simulator evaluation |
| `evaluate <run>` | Fresh simulator evaluation of a completed run |
| `report <run>` | Markdown and JSON report of one run |
| `research sweep \| matrix \| summarize \| reevaluate` | Experiment management |

Run options (`train`, `suite`, `research sweep|matrix`): `--set KEY=JSON`,
`--seed`, `--device auto|cpu|cuda`, `--force`, `--dry-run`, `--no-progress`.
Queue options: `--seeds`, `--execute` (without it only the plan is printed),
`--jobs`, `--gpus 0,1`, `--follow`. A `<run>` is a run directory, a `run_id`,
or a unique directory name.

## `run-all`

```text
[1/4 datasets] 2/4 building simbank/bank/final | stage 3m10s | total 3m10s
[2/4 baselines] ...
[3/4 training] 37/160 runs done (8 running, 0 failed) | stage 41m02s | total 52m15s | ETA 1h45m10s
[4/4 summary] done | stage 2s | total 2h40m01s
```

The training line stays at the top of the screen with a bar per running job;
without a terminal (`nohup ... > log`) it is printed every
`progress_interval_seconds` of the experiment file. The ETA is the mean
duration of the finished runs of each method times the runs left, divided by
`--jobs`. Several jobs may share one GPU. Interrupting and rerunning the same
command resumes: built datasets, current baselines, and completed runs are
reused; failed runs are retried. A dataset of an older schema
([data-schema.md](data-schema.md)) is rebuilt, and runs trained on it are not
reused (the schema version is part of the run identity).

## Queues

`run-all`, `suite`, and `research sweep|matrix` run each job in its own worker
process with its own log and progress file. Completed runs are skipped unless
`--force` is given. Queue state and `suite_results.json` are written to
`artifacts/queues/<timestamp>/`.

## Run lifecycle

`run_manifest.json` records identity (including the representation),
provenance (git commit, Python, torch, devices), and status:

| Status | Meaning |
|---|---|
| `planned`, `queued`, `running` | Not finished |
| `completed` | All artifacts written |
| `failed` | An exception stopped the run (`error` holds it) |
| `interrupted` | Stopped by Ctrl+C or SIGTERM |
| `stale` | Shown by `status` for a `running` run whose process no longer exists |

## Artifact layout

```text
artifacts/
  datasets/<env>/<behavior>/<tier>/{train,val}.npz, metadata.json, episodes_summary.csv
  runs/<env>/<behavior>/<tier>/<method>/<representation>/seed-<n>--<hash>/
      config.json  run_manifest.json  train_history.csv
      checkpoint_best.pt  checkpoint_last.pt        (include the fitted representation)
      eval_summary.json  environment_spec.json  metrics_summary.json
      evaluations/<timestamp>/                       (later re-evaluations)
  queues/<timestamp>/{jobs,logs,progress}/, queue_manifest.json, suite_results.json
  reports/baselines/<dataset>/baseline_summary.json
  reports/summary/<timestamp>/{summary.md,summary.json,runs.csv,by_method.csv,comparison.csv}
  reports/summary/<timestamp>/loss_curves/<dataset>.png
```

`episodes_summary.csv` (one row per evaluation episode) and
`trajectories.csv` (one row per evaluation step) are written only with
`save_trajectories: true` (for example `--set save_trajectories=true`); the
protocol turns them off to keep the 160 runs small.
