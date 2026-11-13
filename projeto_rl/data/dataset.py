"""Loading offline datasets (schema: ``docs/data-schema.md``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from projeto_rl.envs.base import FeatureSpec
from projeto_rl.paths import PROJECT_ROOT
from projeto_rl.utils.io import load_json

REQUIRED_ARRAYS = {
    "observations",
    "next_observations",
    "actions",
    "next_actions",
    "rewards",
    "terminals",
    "timeouts",
    "state_ids",
    "next_state_ids",
    "action_masks",
    "next_action_masks",
    "behavior_logprobs",
    "behavior_policy_ids",
    "episode_ids",
    "timesteps",
}


@dataclass
class OfflineBatch:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    terminal: torch.Tensor
    action_mask: torch.Tensor
    next_action_mask: torch.Tensor


def resolve_dataset_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


class OfflineTransitionDataset:
    """One split (``train.npz`` or ``val.npz``) held in memory."""

    def __init__(self, path: str | Path, *, seed: int | None = None):
        self.path = resolve_dataset_path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        metadata_path = self.path.parent / "metadata.json"
        self.metadata = load_json(metadata_path) if metadata_path.exists() else {}
        with np.load(self.path, allow_pickle=False) as payload:
            missing = sorted(REQUIRED_ARRAYS - set(payload.files))
            if missing:
                raise ValueError(f"Offline dataset {self.path} is missing required arrays: {missing}")
            self.data = {key: payload[key] for key in payload.files}
        self.size = int(len(self.data["actions"]))
        if self.size == 0:
            raise ValueError(f"Offline dataset {self.path} is empty")
        self.rng = np.random.default_rng(seed)
        self._index_episodes()

    def _index_episodes(self) -> None:
        episode_ids = self.data["episode_ids"]
        #: Transition indices of every episode, in timestep order.
        self.episode_indices = [
            indices[np.argsort(self.data["timesteps"][indices], kind="stable")]
            for indices in (np.flatnonzero(episode_ids == episode) for episode in np.unique(episode_ids))
        ]

    def __len__(self) -> int:
        return self.size

    @property
    def num_episodes(self) -> int:
        return len(self.episode_indices)

    @property
    def obs_dim(self) -> int:
        return int(self.data["observations"].shape[1])

    @property
    def num_actions(self) -> int:
        return int(self.data["action_masks"].shape[1])

    @property
    def observations(self) -> np.ndarray:
        return self.data["observations"]

    @property
    def next_observations(self) -> np.ndarray:
        return self.data["next_observations"]

    @property
    def features(self) -> tuple[FeatureSpec, ...]:
        """Column types recorded at collection time (``metadata.json``)."""

        try:
            payload = self.metadata["observation"]["features"]
        except KeyError as exc:
            raise ValueError(f"{self.path.parent / 'metadata.json'} has no observation.features") from exc
        features = tuple(FeatureSpec.from_dict(item) for item in payload)
        if len(features) != self.obs_dim:
            raise ValueError(f"{self.path}: {len(features)} declared features but obs_dim={self.obs_dim}")
        return features

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    def apply_representation(self, representation) -> None:
        """Replace ``observations`` and ``next_observations`` by their represented vectors, in memory."""

        for key in ("observations", "next_observations"):
            self.data[key] = representation.transform(self.data[key])

    def keep_episode_fraction(self, fraction: float, *, seed: int) -> None:
        """Keep ``round(fraction * num_episodes)`` whole episodes, in memory.

        The kept episodes are the first ones of a shuffle that depends only on
        ``seed`` and the number of episodes, so a smaller fraction keeps a
        subset of what a larger one keeps. Transitions stay in storage order.
        """

        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"The episode fraction must be in (0, 1], got {fraction}")
        count = round(fraction * self.num_episodes)
        if count == 0:
            raise ValueError(f"{self.path}: a fraction of {fraction} keeps none of its {self.num_episodes} episodes")
        order = np.random.default_rng(seed).permutation(self.num_episodes)
        kept = np.sort(np.concatenate([self.episode_indices[episode] for episode in order[:count]]))
        self.data = {key: values[kept] for key, values in self.data.items()}
        self.size = len(kept)
        self._index_episodes()

    def mean_abs_reward(self) -> float:
        """Mean ``|reward|`` per transition, the training reward unit (fitted on the train split)."""

        scale = float(np.abs(self.data["rewards"]).mean())
        if not scale > 0.0:
            raise ValueError(f"{self.path}: every reward is zero, so the reward scale is undefined")
        return scale

    def scale_rewards(self, scale: float) -> None:
        """Divide ``rewards`` by ``scale``, in memory."""

        self.data["rewards"] = (self.data["rewards"] / np.float32(scale)).astype(np.float32)

    def _batch(self, indices: np.ndarray, device) -> OfflineBatch:
        def tensor(key: str, dtype) -> torch.Tensor:
            return torch.as_tensor(self.data[key][indices], dtype=dtype, device=device)

        return OfflineBatch(
            obs=tensor("observations", torch.float32),
            action=tensor("actions", torch.int64),
            reward=tensor("rewards", torch.float32),
            next_obs=tensor("next_observations", torch.float32),
            terminal=tensor("terminals", torch.float32),
            action_mask=tensor("action_masks", torch.float32),
            next_action_mask=tensor("next_action_masks", torch.float32),
        )

    def sample(self, batch_size: int, device: str | torch.device = "cpu") -> OfflineBatch:
        """Uniform transitions with replacement (the training batch)."""

        return self._batch(self.rng.integers(0, self.size, size=int(batch_size)), device)

    def iter_transition_batches(self, batch_size: int, *, device: str | torch.device = "cpu"):
        """Every transition exactly once, in storage order (validation)."""

        for start in range(0, self.size, int(batch_size)):
            yield self._batch(np.arange(start, min(start + int(batch_size), self.size)), device)
