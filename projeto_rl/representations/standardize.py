"""Per-column z-score with the train split's mean and standard deviation.

It is the simplest fitted representation and exists to exercise the whole
pipeline (fit on train only, freeze, checkpoint, deploy) end to end.
"""

from __future__ import annotations

import numpy as np

from projeto_rl.representations import register_representation
from projeto_rl.representations.base import Representation


@register_representation
class Standardize(Representation):
    name = "standardize"

    def fit(self, train, *, seed: int) -> None:
        del seed
        obs = np.asarray(train.observations, dtype=np.float64)
        mean = obs.mean(axis=0)
        std = obs.std(axis=0)
        constant = [feature.name for feature, value in zip(self.features, std) if value == 0.0]
        if constant:
            raise ValueError(f"Cannot standardize constant train columns: {constant}")
        self.mean, self.std = mean, std

    def _transform(self, obs):
        return (obs - self.mean) / self.std

    @property
    def output_dim(self) -> int:
        return self.input_dim

    def fitted_state(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    def load_fitted_state(self, state) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.std = np.asarray(state["std"], dtype=np.float64)
        if self.mean.shape != (self.input_dim,) or self.std.shape != (self.input_dim,):
            raise ValueError("Standardize state does not match the number of features")
