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
    name: ClassVar[str]  # value of `representation` in configs

    def __init__(self, features, config): ...  # features: tuple[FeatureSpec, ...]
    def fit(self, train, *, seed: int) -> None: ...  # train: OfflineTransitionDataset
    def _transform(self, obs: np.ndarray) -> np.ndarray: ...  # [N, d] float32 -> [N, k]
    @property
    def output_dim(self) -> int: ...  # k
    def fitted_state(self) -> dict: ...  # what fit learned (JSON values or tensors)
    def load_fitted_state(self, state: dict) -> None: ...
```

The base class checks the input shape, casts the output to `float32`, checks
the output shape, and stores the feature types with the fitted state.
`self.features[i].kind` is one of `numeric`, `binary`, `ordinal`,
`categorical`, `cyclic`; the last three have `cardinality` and take integer
codes `0..cardinality-1`. `self.config` is the full run config, so a
representation can read its own hyperparameters from it.

## Shipped representations

| Name | Output | What it fits on the train split |
|---|---|---|
| `raw` | the observation, `k = d` | nothing |
| `standardize` | per-column z-score, `k = d` | column means and standard deviations |
| `spr` | a learned latent, `k = spr_latent_dim` | an encoder, from the logged transitions |
| `tjepa` | one token per column, `k = d * tjepa_hidden_dim` | column statistics and an encoder, from the observations |
| `tabpfn` | the row embedding of a pretrained model, `k = 1024` for TabPFN-3.5 (`k = 192` for the TabPFN-2 fallback) | nothing is trained; a context of logged rows and the embedding statistics are kept |
| `tdjepa` | a learned latent on a sphere, `k = tdjepa_state_dim` | column statistics and an encoder, from the logged transitions |

### `spr`

Self-Predictive Representations (Schwarzer et al., "Data-Efficient
Reinforcement Learning with Self-Predictive Representations", ICLR 2021,
arXiv:2007.05929), used as frozen pretraining. An encoder maps the raw
observation (no input scaling) to a latent; a transition model predicts the next latent from the
latent and the logged action; the target is the next observation encoded by
an exponential moving average of the encoder; the loss is the negative cosine
similarity between the two after a projection and a linear prediction head.
After `spr_train_steps` gradient steps only the encoder is kept, and its
latent is z-scored per dimension with train statistics.

Differences from the paper, also listed in the class docstring: the SPR loss
is the only objective (the paper adds it to Rainbow's loss) and the encoder
is frozen afterwards; one prediction step instead of five; MLPs instead of
convolutions, with the one-hot action concatenated to the latent; a
projection layer of its own, since there is no Q-learning head to borrow one
from. The target coefficient (0.99) and the dropout (0.5) are the paper's
settings for training without data augmentation.

Its hyperparameters are the `spr_*` keys of `configs/protocol.json`. Every
run refits its own encoder with the run's seed, on the GPU when one is
visible; the frozen encoder is then applied on the CPU.

### `tjepa`

T-JEPA (Thimonier et al., "T-JEPA: Augmentation-Free Self-Supervised Learning
for Tabular Data", ICLR 2025, arXiv:2410.05016), used as frozen pretraining.
It learns from single rows: actions, rewards, and next observations are not
used. Numeric columns are standardized with train statistics and `ordinal`,
`categorical`, and `cyclic` columns are one-hot; every column has its own
linear layer that turns it into a token. A context encoder (a transformer)
sees a random subset of the columns of a row; a target encoder, an
exponential moving average of the context encoder, sees all of them; a
smaller transformer predicts the target encoder's tokens of other columns
from the context tokens; the loss is the squared distance between the two. A
learned [REG] token is appended to the input of both encoders and dropped
from their outputs; the paper reports that training collapses without it.
After `tjepa_train_steps` gradient steps the context encoder is kept, and the
representation of a row is its `d` column tokens side by side.

Differences from the paper, also listed in the class docstring: a fixed
number of steps instead of epochs with model selection by a linear probe on
labels; the token matrix is flattened, where the paper's downstream models
learn a projection of it together with the task; `ordinal` and `cyclic`
columns are treated as categorical; the index and feature-type embeddings
are the bias of each column's layer; no dropout, Adam, no warm-up. Where the
released code (https://github.com/jose-melo/t-jepa) differs from the paper,
the paper is followed.

Its hyperparameters are the `tjepa_*` keys of `configs/protocol.json`: the
transformer sizes and the mask shares are the released code's defaults, the
batch size, the number of target masks, and the moving-average schedule are
the paper's. Every run refits its own encoder with the run's seed, on the GPU
when one is visible.

### `tabpfn`

TabPFN (Hollmann et al., "Accurate predictions on small data with a tabular
foundation model", Nature 2025; the version in `configs/protocol.json` is
TabPFN-3.5, technical report arXiv:2609.17895) is a transformer pretrained on
synthetic tables that predicts the target of a row from labelled rows given
in context. It is not self-supervised and nothing is trained in a run: `fit`
draws `tabpfn_context_size` rows of the train split, gives them to the
pretrained classifier with the logged action as the label, and keeps the
conditioned model. The representation of an observation is the vector the
classification head reads for that observation as a query row
(`get_embeddings` of the `tabpfn` package), z-scored per dimension with
train statistics.

Choices made here, also listed in the class docstring: the logged action is
the target (rewards and next observations are not used); the context is a
uniform sample of the train split, because a call to the model costs more
the larger the context and the simulator needs one call per step (about
13 ms with 2,000 context rows on the reference GPU, 32 ms with 10,000,
whatever the number of episodes evaluated side by side); one estimator; 32-bit inference instead of the package's mixed
precision, so that a row has the same embedding alone and in a batch;
`categorical` columns are declared to the model and the others are read as
numbers.

The checkpoint holds the context rows, their labels, the seed, the z-score
statistics, and the version that was conditioned. The pretrained weights are not in it: they stay in the
`tabpfn` cache and the conditioned model is rebuilt from the context at
evaluation time, on the run's device. The embedding differs slightly between
the CPU and the GPU (up to 0.08 in z-score units, 0.001 on average, measured
on the four final datasets), so a checkpoint should be evaluated on the kind
of device it was trained on.

The code of the `tabpfn` package is Apache-2.0. The weights of TabPFN-2.5 and
later, including 3.5, are under a non-commercial license (research and
internal evaluation are allowed) that each user accepts once at
https://ux.priorlabs.ai before the first download; on a machine without a
browser, set `TABPFN_TOKEN` to the API key of that account for the first run.
The tests use the TabPFN-2 weights, which download without a license.

Fallback: when the package refuses the weights of `tabpfn_version` because
the license was not accepted (`TabPFNLicenseError` or
`TabPFNHuggingFaceGatedRepoError`), `fit` emits a warning and conditions
`tabpfn_fallback_version` (`v2`) on the same context instead. Any other
failure is raised. TabPFN-2 is another model, with a 192-dimensional
embedding, so a run that fell back is not comparable with a TabPFN-3.5 run:
it is identified by the warning in the run log, by
`representation_output_dim = 192` in the run summary, and by `version` in the
checkpoint's representation state. Evaluation loads the version stored in the
checkpoint and never falls back.

### `tdjepa`

TD-JEPA (Bagatella et al., "TD-JEPA: Latent-predictive Representations for
Zero-Shot Reinforcement Learning", ICLR 2026, arXiv:2510.00739), used as
frozen pretraining of a state encoder. It learns from the logged transitions
without rewards. A family of policies is indexed by a task vector `z`; a
state encoder `phi` and a predictor `T_phi(phi(s), a, z)` are trained so that
the predictor gives the discounted sum of the task latents `psi(s_t)` that
the policy of `z` visits after `(s, a)` (its successor features), and a task
encoder `psi` with its own predictor is trained by the same loss with the two
encoders swapped. The sum is learned from single transitions through its
Bellman equation, with target copies of the four networks, and an
orthonormality regularizer keeps both latents from collapsing. Unlike `spr`,
the encoder is asked for what decides the long-term future under many
policies, not only the next step under the logged one.

The representation is `phi` of the z-scored observation: a vector of
`tdjepa_state_dim` entries on the sphere of radius `sqrt(k)`, so its entries
have unit mean square. The task encoder and the predictors are discarded.

Adaptations, also listed in the class docstring: actions are discrete, so
the policy of each task is the exact argmax of `T_phi(phi(s), a, z) . z`
over the valid actions instead of a trained actor; the encoder is frozen and
only it is kept; `tdjepa_train_steps` gradient steps instead of the paper's
one to three million, so the temporal-difference targets are far from
converged when the encoder is frozen; no behavior-cloning term on the
policies (the paper adds one, defined for continuous actions, on datasets
with narrow action coverage); every column is z-scored first, as
`standardize` does, where the paper feeds observations unscaled.

Its hyperparameters are the `tdjepa_*` keys of `configs/protocol.json`, all
of them the defaults of the released code
(https://github.com/facebookresearch/td_jepa, CC BY-NC 4.0, read for the
details the paper leaves out; nothing is copied from it). They match the
paper's proprioceptive DeepMind Control setting except the state dimension
(50; the paper's table lists 256), the depth of the state encoder (two
hidden layers; the paper tunes it per benchmark), and the moving-average
coefficient (0.995, the paper's OGBench value). Every run refits its own
encoder with the run's seed, on the GPU when one is visible: about 100 to
170 seconds per fit alone on the reference GPU. Choosing the next action
evaluates every action of every row; only the argmax of that pass is used,
so on a GPU it runs in half precision.

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
