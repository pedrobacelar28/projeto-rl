# Representations

A representation maps raw tabular observations `[N, d]` to vectors `[N, k]`
before any algorithm sees them. The algorithms themselves are unchanged: a run
with `representation = "<name>"` trains the same networks, on the same data,
with the same hyperparameters, only on transformed inputs.

## Life cycle of a run

1. `training.py` builds the representation of `config["representation"]`
   with the column types of the dataset (`metadata.json`, `observation.features`).
2. `fit(train, seed=...)` sees the **train split only**: observations,
   actions, rewards, next observations, episode ids. Never the validation split
   or the simulator.
3. The fitted representation is frozen and applied to the train and
   validation splits in memory; the stored dataset stays raw.
4. Its state (`state_dict()`) is saved in every checkpoint, together with the
   network input size.
5. At evaluation, `load_policy` rebuilds it from the checkpoint (no refit) and
   wraps the trained policy in `RepresentationPolicy`, which transforms each
   raw simulator observation before acting.

## Contract

`projeto_rl/representations/base.py`:

```python
class Representation:
    name: ClassVar[str]                                  # value of `representation` in configs
    def __init__(self, features, config): ...            # features: tuple[FeatureSpec, ...]
    def fit(self, train, *, seed: int) -> None: ...      # train: OfflineTransitionDataset
    def _transform(self, obs: np.ndarray) -> np.ndarray: ...  # [N, d] float32 -> [N, k]
    @property
    def output_dim(self) -> int: ...                     # k
    def fitted_state(self) -> dict: ...                  # what fit learned (JSON values or arrays)
    def load_fitted_state(self, state: dict) -> None: ...
```

The base class checks the input shape, casts the output to `float32`, checks
the output shape, and stores the feature types with the fitted state.
`self.features[i].kind` is one of `numeric`, `binary`, `ordinal`,
`categorical`, `cyclic`; the last three have `cardinality` and take integer
codes `0..cardinality-1`. `self.config` is the full run config, so a
representation can read its own hyperparameters from it.

## Adding one

1. Write `projeto_rl/representations/<name>.py`:

   ```python
   @register_representation
   class OneHot(Representation):
       name = "onehot"
       def fit(self, train, *, seed): ...
       def _transform(self, obs): ...
       @property
       def output_dim(self): ...
       def fitted_state(self): ...
       def load_fitted_state(self, state): ...
   ```

2. Import it at the bottom of `projeto_rl/representations/__init__.py`.
3. If it has hyperparameters, put their defaults in `configs/protocol.json`
   (prefixed with the representation name), never in the code.
4. Add tests against a known truth in `tests/test_representations.py`
   (values computed by hand on a small array, train-only fitting, state round
   trip); `tests/test_algorithms.py` then trains every method with it on every
   environment.
5. Add its name to `representations` in `configs/experiments/*.json` and run
   `projeto-rl run-all configs/experiments/final.json --gpus 0 --jobs <N>`: the
   finished `standardize` runs are reused, only the new representation trains,
   and the summary compares it with `standardize` (the experiment's
   `reference_representation`) on the same seeds.
