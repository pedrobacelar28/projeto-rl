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
- SPR: the cosine loss and the target moving average on hand-computed
  values; the transform equals a forward pass done by hand from the stored
  weights on the raw observation; on a deterministic system the loss reaches its floor of -1 and the
  latent still decodes every column; the same seed gives the same encoder and
  the fit leaves the global generators untouched.
- T-JEPA: the squared-error loss, the sine-cosine positions, and the mask
  sizes on hand-computed values; target masks never overlap the context; the
  transform equals a transformer layer written in numpy, fed with tokens
  built by hand from the stored weights; on a table whose columns determine
  each other the loss falls from 1 (the zero predictor against
  layer-normalized targets) and the representation still decodes every
  numeric column; seeding and the global generators as for SPR.
- TabPFN: the transform equals the z-scored embedding returned by the
  `tabpfn` package driven directly with the context rows the seed draws;
  other labels on the same rows give another embedding; one row at a time
  gives the embedding of the batch; the conditioned model is rebuilt exactly
  from a checkpoint file; when the package refuses the configured version
  for lack of a license the result is the embedding of the fallback version,
  with a warning, and any other failure is raised. These tests use the
  TabPFN-2 weights (no license, downloaded on the first run) and are marked
  `slow`.
- TD-JEPA: the prediction loss, the orthonormality regularizer, the
  projection on the sphere, and the greedy action over valid actions on
  hand-computed values; task vectors lie on the sphere and mix goals with
  uniform draws in the configured share; the temporal-difference targets
  equal the paper's Bellman equation evaluated row by row and action by
  action, with masked actions excluded and no future after a terminal
  row; the transform equals a forward pass done by hand from the stored
  weights on z-scored columns; on a deterministic system the prediction
  loss falls from the spread of the latents to a small fraction of it, the
  second-moment matrix of the latent is close to the identity (the
  regularizer's optimum), and without the regularizer one direction holds
  most of the norm; seeding and the global generators as for SPR.
- Evaluation: on every environment, episodes run side by side are the
  episodes of a sequential evaluation, row by row; only stateless policies
  acting greedily are run side by side.
- Statistics: normalized score, paired difference and its Student-t interval,
  the interquartile mean, and the queue ETA on hand-computed cases.
- Training targets: the TD target bootstraps through a timeout but not
  through a terminal (hand-computed); the reward scale is the train split's
  mean `|reward|` and ignores the validation split; every environment logs
  its time limit as a timeout and its true ends as terminals.
- Every algorithm trains, checkpoints, and acts within the mask on every
  environment with every representation.
