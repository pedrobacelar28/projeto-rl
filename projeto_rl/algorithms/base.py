"""The interface every offline algorithm implements.

An algorithm owns its networks and optimizer, turns a batch into a loss, and
builds the policy that is deployed from a checkpoint. The shared loop in
:mod:`projeto_rl.training` handles batching, validation, early stopping,
checkpoint selection, and logging, so a new method is usually one file with
``build``, ``loss``, and ``make_policy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Iterator

import torch
import torch.nn as nn

from projeto_rl.policies import Policy


@dataclass(frozen=True)
class DataSpec:
    """Dimensions an algorithm needs to build its networks."""

    obs_dim: int
    num_actions: int


def td_target(batch, next_value: torch.Tensor, gamma: float) -> torch.Tensor:
    """One-step target ``r + gamma * V(s') * (1 - terminal)``.

    Only true terminal states stop the bootstrap; a step cut by a time limit
    (``timeouts`` in the dataset) still bootstraps from ``s'``.
    """

    return batch.reward + float(gamma) * next_value * (1.0 - batch.terminal)


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Detached CPU copy, so checkpoint snapshots never duplicate GPU memory."""

    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


class OfflineAlgorithm:
    #: Registry key, stored as ``algo`` in configs.
    name: ClassVar[str]
    #: Public method id used by the CLI and reports.
    method_id: ClassVar[str]
    description: ClassVar[str] = ""
    #: What the early-stopping monitor (validation loss) measures.
    validation_objective: ClassVar[str] = "validation loss"
    #: Environment action types this method supports (``configs/envs/*.json``).
    action_types: ClassVar[tuple[str, ...]] = ("discrete",)

    def __init__(self, config: dict[str, Any], spec: DataSpec, device: torch.device):
        self.config = config
        self.spec = spec
        self.device = device
        self.model: nn.Module = self.build().to(device)
        self.optimizer = self.make_optimizer()

    # -- construction ---------------------------------------------------
    def build(self) -> nn.Module:
        """Return the module whose weights are checkpointed."""

        raise NotImplementedError

    def make_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=float(self.config["learning_rate"]))

    # -- optimization ---------------------------------------------------
    def batches(self, dataset) -> Iterator[Any]:
        batch_size = int(self.config["offline_batch_size"])
        while True:
            yield dataset.sample(batch_size, device=self.device)

    def batch_size(self, batch) -> int:
        """Number of logged transitions in a batch (for sampling budgets)."""

        return int(batch.action.shape[0])

    def loss(self, batch) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def update(self, batch, step: int) -> dict[str, float]:
        self.model.train()
        loss, info = self.loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        max_grad_norm = float(self.config.get("max_grad_norm", 0.0))
        if max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()
        self.after_update(step)
        return {"loss": float(loss.item()), **info}

    def after_update(self, step: int) -> None:
        """Hook for target-network updates."""

    @torch.no_grad()
    def validate(self, dataset) -> tuple[float, dict[str, float]]:
        """Transition-weighted mean of :meth:`loss` over the validation split."""

        self.model.eval()
        total, count, sums = 0.0, 0, {}
        for batch in dataset.iter_transition_batches(int(self.config["offline_batch_size"]), device=self.device):
            loss, info = self.loss(batch)
            size = self.batch_size(batch)
            total += float(loss.item()) * size
            count += size
            for key, value in info.items():
                sums[key] = sums.get(key, 0.0) + float(value) * size
        if count == 0:
            raise RuntimeError(f"Validation dataset {dataset.path} produced no batches")
        return total / count, {key: value / count for key, value in sums.items()}

    # -- checkpoints and deployment ---------------------------------------
    @classmethod
    def from_checkpoint(
        cls, config: dict[str, Any], spec: DataSpec, device: torch.device, checkpoint: dict[str, Any]
    ) -> "OfflineAlgorithm":
        algorithm = cls(config, spec, device)
        algorithm.load_state_dict(checkpoint)
        return algorithm

    def state_dict(self) -> dict[str, Any]:
        return {"model_state_dict": cpu_state_dict(self.model)}

    def load_state_dict(self, checkpoint: dict[str, Any]) -> None:
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def make_policy(self) -> Policy:
        raise NotImplementedError
