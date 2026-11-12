"""TabPFN: row embeddings of a pretrained tabular foundation model conditioned on logged train rows."""

from __future__ import annotations

import warnings

import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabpfn.errors import TabPFNHuggingFaceGatedRepoError, TabPFNLicenseError

from projeto_rl.representations import register_representation
from projeto_rl.representations.base import Representation
from projeto_rl.utils.torch_utils import get_torch_device

TRANSFORM_CHUNK = 16384  # rows embedded at once by transform; bounds memory only


@register_representation
class TabPFN(Representation):
    """Frozen TabPFN embeddings of the observation, given a context of logged (observation, action) rows.

    TabPFN (Hollmann et al., "Accurate predictions on small data with a
    tabular foundation model", Nature 2025; the checkpoint used here is the
    one of the TabPFN-3.5 technical report, arXiv:2609.17895) is a transformer
    pretrained on synthetic tables that predicts the target of a new row from
    labelled rows given in context, with no gradient step. Its row embedding
    is the output of the in-context transformer for a query row, the vector
    its classification head reads (``get_embeddings`` of the ``tabpfn``
    package, used as in "A Closer Look at TabPFN v2", arXiv:2502.17361).

    Nothing is trained here. ``fit`` draws ``tabpfn_context_size`` rows of the
    train split, hands them to the pretrained classifier with the logged
    action as the label, and keeps the resulting conditioned model. The
    representation of an observation is its embedding as a query row against
    that context.

    Choices made for this project:

    - The model is supervised, so it needs a target; the logged action is
      used because it is what every method here has to predict. The reward
      and the next observation never enter the context.
    - The context is a uniform sample of the train split without replacement,
      because the cost of a call grows with the context and simulator
      evaluation makes one call per step. Train rows that are in
      the context are embedded like any other row, so they see their own
      label in context.
    - One estimator, so there is no ensemble to aggregate and the embedding
      is the one of a single forward pass.
    - Inference runs in 32-bit floats instead of the package default
      (mixed precision on the GPU): with mixed precision the embedding of a
      row changed by up to 2e-2 between a batch and a single-row call, which
      is the difference between training and deployment.
    - Columns of kind ``categorical`` are declared as such; the others are
      read as numbers, so ordinal and cyclic codes keep their order.
    - The embedding is z-scored per dimension with train-split statistics,
      as the ``spr`` latent is: its raw scale is uneven (a standard deviation
      near 0.03 in most dimensions and above 1 in a few), and the downstream
      networks should see inputs of unit scale.

    The weights of the 2.5 and later versions are released under a
    non-commercial license that has to be accepted once per account before
    they download. Documented fallback: when the package refuses the weights
    of ``tabpfn_version`` for that reason, ``fit`` warns and conditions
    ``tabpfn_fallback_version`` instead (TabPFN-2 in the protocol, whose
    weights download without that step). It is another model, with another
    embedding width, so a run that fell back is not comparable with one that
    did not. Any other failure is raised as it is.

    The checkpoint stores the context rows, their labels, the seed, the
    z-score statistics, and the version that was conditioned; the pretrained
    weights stay in the ``tabpfn`` cache and are loaded again at evaluation
    time on the run's device. Evaluation loads the stored version and never
    falls back.
    """

    name = "tabpfn"

    def __init__(self, features, config):
        super().__init__(features, config)
        self.version = ModelVersion(self._setting("tabpfn_version"))
        self.fallback_version = ModelVersion(self._setting("tabpfn_fallback_version"))
        self.context_size = int(self._setting("tabpfn_context_size"))
        if self.context_size <= 0:
            raise ValueError("tabpfn_context_size must be > 0")
        self.categorical = [index for index, feature in enumerate(self.features) if feature.kind == "categorical"]

    def _setting(self, key: str):
        if key not in self.config:
            raise ValueError(f"The tabpfn representation needs {key!r} in the run config (configs/protocol.json)")
        return self.config[key]

    def _condition(self, context: np.ndarray, labels: np.ndarray, seed: int, version: ModelVersion) -> None:
        if context.shape != (len(labels), self.input_dim):
            raise ValueError(f"TabPFN context of shape {context.shape} does not match {len(labels)} labels")
        if len(np.unique(labels)) < 2:
            raise ValueError("The TabPFN context needs at least two distinct logged actions")
        model = TabPFNClassifier.create_default_for_version(
            version,
            n_estimators=1,
            # Processes the context once; each later call only runs the query rows.
            fit_mode="fit_with_cache",
            device=str(get_torch_device()),
            random_state=seed,
            inference_precision=torch.float32,
            categorical_features_indices=self.categorical,
        )
        model.fit(context, labels)
        self.model, self.context, self.labels, self.seed, self.used_version = model, context, labels, seed, version

    def _embed(self, obs: np.ndarray) -> np.ndarray:
        chunks = [
            np.asarray(self.model.get_embeddings(obs[start : start + TRANSFORM_CHUNK], data_source="test")).reshape(
                len(obs[start : start + TRANSFORM_CHUNK]), -1
            )
            for start in range(0, len(obs), TRANSFORM_CHUNK)
        ]
        return np.concatenate(chunks)

    def fit(self, train, *, seed: int) -> None:
        observations = np.asarray(train.observations, dtype=np.float32)
        if self.context_size > len(observations):
            raise ValueError(
                f"tabpfn_context_size ({self.context_size}) exceeds the train split ({len(observations)} rows)"
            )
        rows = np.sort(np.random.default_rng(seed).choice(len(observations), size=self.context_size, replace=False))
        context, labels = observations[rows], np.asarray(train.data["actions"], dtype=np.int64)[rows]
        try:
            self._condition(context, labels, seed, self.version)
        except (TabPFNLicenseError, TabPFNHuggingFaceGatedRepoError) as error:
            warnings.warn(
                f"The TabPFN {self.version.value} weights need a license that was not accepted on this machine; "
                f"using tabpfn_fallback_version ({self.fallback_version.value}) instead. This run is not "
                f"comparable with runs of {self.version.value}. Reason: {error}",
                stacklevel=2,
            )
            self._condition(context, labels, seed, self.fallback_version)
        embedding = self._embed(observations).astype(np.float64)
        if not np.isfinite(embedding).all():
            raise ValueError("TabPFN returned non-finite embeddings on the train split")
        mean, std = embedding.mean(axis=0), embedding.std(axis=0)
        constant = np.flatnonzero(std == 0.0).tolist()
        if constant:
            raise ValueError(f"TabPFN embedding dimensions {constant} are constant on the train split")
        self.mean, self.std = mean, std

    def _transform(self, obs):
        return (self._embed(obs) - self.mean) / self.std

    @property
    def output_dim(self) -> int:
        return len(self.mean)

    def fitted_state(self):
        return {
            "context": torch.from_numpy(self.context),
            "labels": torch.from_numpy(self.labels),
            "seed": self.seed,
            "version": self.used_version.value,
            "mean": torch.from_numpy(self.mean),
            "std": torch.from_numpy(self.std),
        }

    def load_fitted_state(self, state) -> None:
        self._condition(
            state["context"].numpy(), state["labels"].numpy(), int(state["seed"]), ModelVersion(state["version"])
        )
        self.mean, self.std = state["mean"].numpy(), state["std"].numpy()
