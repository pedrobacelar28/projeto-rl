"""The identity representation: algorithms see the observation exactly as the environment emits it."""

from __future__ import annotations

from projeto_rl.representations import register_representation
from projeto_rl.representations.base import Representation


@register_representation
class Raw(Representation):
    name = "raw"

    def fit(self, train, *, seed: int) -> None:
        del train, seed

    def _transform(self, obs):
        return obs

    @property
    def output_dim(self) -> int:
        return self.input_dim

    def fitted_state(self):
        return {}

    def load_fitted_state(self, state) -> None:
        if state:
            raise ValueError(f"The raw representation has no fitted state, got {sorted(state)}")
