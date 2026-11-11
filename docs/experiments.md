# Experiments

## Protocol of one run

1. Load the train and validation splits of a dataset (collected once in the
   simulator by the dataset's behavior policy).
2. Fit the run's representation on the train split and apply it to both
   splits ([representations.md](representations.md)).
3. Divide every reward by the train split's mean `|reward|`, in memory, so
   the Q-values of every environment have a comparable scale (an Industrial
   Benchmark step costs about 200, a SimBank step pays 0.7 on average). The
   scale is recorded as `reward_scale` in the run's `train_summary`; the
   evaluation below reports the unscaled return.
4. Train for `offline_train_steps` gradient steps; every
   `offline_log_interval` steps compute the validation loss; keep the
   checkpoint with the lowest validation loss; stop early after
   `early_stopping_patience` checks without improvement.
5. Deploy the selected checkpoint greedily in the simulator for
   `eval_episodes` episodes with seeds `eval_seed_base + i`, on the
   evaluation entities (`purpose="evaluate"`), and record the mean return.

The simulator never enters steps 1-4.

Step 5 runs `eval_parallel_episodes` episodes side by side, each in its own
environment and with its own seed, and represents their observations in one
call per step. The episodes are the ones a sequential evaluation produces
(tested on every environment); only the time changes, because a learned
representation costs about the same for one observation as for a batch: an
Industrial Benchmark step with `tabpfn` went from 13.6 ms to 0.47 ms. The
random and behavior policies, which keep per-episode state or draw their
actions, are still evaluated one episode at a time.

**Time limits are not terminal states.** A transition cut by the
environment's time limit is logged in `timeouts`, not in `terminals`, and the
TD target still bootstraps from the next observation there (Pardo et al.,
"Time Limits in Reinforcement Learning", ICML 2018). This matters for the
Industrial Benchmark (every episode is cut after 250 steps, with γ = 0.99)
and EpiCare (episodes that reach the 8th visit); SalesPromotion and SimBank
end in true terminal states (the day or decision index is observed).

## Normalized score

`projeto-rl baselines <dataset>` evaluates two reference policies of the dataset's
environment with the same evaluation seeds and episode count as the runs, and
stores their mean returns in `artifacts/reports/baselines/<dataset>/`:

- `low`: the random policy (`normalization.low` in `configs/envs/<env>.json`);
- `high`: the behavior policy of the dataset, with its exploration ε.

Both are sampled (not greedy), as they were when they produced the data. The
score of a run is

```
score = 100 * (R - R_low) / (R_high - R_low)
```

so scores of different environments can be averaged; raw returns are never
averaged across environments.

## Comparison with the reference representation

Each experiment names a `reference_representation` (`standardize`). For each
dataset, method, and other representation, the summary pairs every seed with
the reference run of the same seed and reports the mean difference of the
normalized score, a 95% Student-t confidence interval over seeds, and the
number of seeds where the representation wins. It also reports, per
representation, the interquartile mean (IQM) of the normalized score per
dataset, over the homogeneous environments, over the heterogeneous ones, and
over all. The IQM averages the middle 50% of the runs, so one diverged run
cannot dominate a group (Agarwal et al., "Deep Reinforcement Learning at the
Edge of the Statistical Precipice", NeurIPS 2021).

With 3 seeds a single comparison has wide intervals (t = 4.30); the
aggregated rows carry more weight.

## Loss curves

The summary also writes `loss_curves/<dataset>.png`: one row of panels per
method, the training loss on the left and the validation loss on the right,
one line per seed colored by representation, and a dot on the checkpoint
selected by the validation loss. The y axis is logarithmic when every loss in
the row is positive.

## Experiment files

`configs/experiments/<name>.json`:

```json
{
  "description": "...",
  "datasets": ["epicare/soc/final", "..."],
  "methods": ["bc", "dqn", "iql", "cql"],
  "representations": ["standardize"],
  "reference_representation": "standardize",
  "seeds": [0, 1, 2],
  "progress_interval_seconds": 30
}
```

`dev.json` is the smoke test (small datasets, one seed); `final.json` is the
experiment. `projeto-rl run-all <file>` runs datasets, baselines, training, and the
summary; `projeto-rl research matrix <file> --execute` runs only the training, and
`projeto-rl research summarize <file>` rebuilds the summary of the experiment's
runs (only those, so runs of an older protocol never enter the tables).
