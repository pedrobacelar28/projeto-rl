# Development

## Tests

```bash
pytest                          # full suite; uses the GPU when one is visible
pytest -m "not slow"            # quick check, run by CI on every push
CUDA_VISIBLE_DEVICES="" pytest  # CPU only, for when the GPU is busy
pre-commit run --all-files      # ruff check and format
```

**Device.** Tests that train use `PROJETO_RL_DEVICE` (default `auto`: CUDA when
available, otherwise CPU). `PROJETO_RL_DEVICE=cpu` or `PROJETO_RL_DEVICE=cuda` forces
one of them.

**Slow tests.** Mark a test `@pytest.mark.slow` when it takes longer than
10 s on its own (today only the end-to-end `run-all` test).
`--strict-markers` is on, so a misspelled marker is an error.

**What is tested against a known truth.**

- Environments: every rollout matches its declared column types; resets with
  a seed reproduce trajectories; the Industrial Benchmark adapter reproduces
  the vendored simulator step by step; the SimBank adapter reproduces the
  source's online loop and pays the case outcome; the SalesPromotion
  observable update matches `_get_next_state` on hand-made user answers.
- The SalesPromotion port was also checked once against the original NeoRL
  code (with stub `gym` and `neorl` modules): identical costs, GMV, and
  states on 200 users over 5 days (see `third_party/sales-promotion/NOTICE`).
- Representations: identity, z-score values by hand, train-only fitting,
  state round trip, deployment through `RepresentationPolicy`.
- Statistics: normalized score, paired difference and its Student-t interval,
  the interquartile mean, and the queue ETA on hand-computed cases.
- Training targets: the TD target bootstraps through a timeout but not
  through a terminal (hand-computed); the reward scale is the train split's
  mean `|reward|` and ignores the validation split; every environment logs
  its time limit as a timeout and its true ends as terminals.
- Every algorithm trains, checkpoints, and acts within the mask on every
  environment with every representation.
