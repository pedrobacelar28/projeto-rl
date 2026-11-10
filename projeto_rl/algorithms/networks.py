"""Feed-forward networks and the two generic policies that read them."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from projeto_rl.policies import Policy, greedy

MASKED_LOGIT = -1e9


def build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers += [nn.Linear(last_dim, hidden_dim), nn.ReLU()]
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def hidden_dims(config: dict) -> list[int]:
    width = int(config.get("policy_hidden_dim", 256))
    layers = int(config.get("policy_hidden_layers", 2))
    if width <= 0 or layers <= 0:
        raise ValueError("policy_hidden_dim and policy_hidden_layers must be > 0")
    return [width] * layers


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
    if action_mask is None:
        return logits
    return logits.masked_fill(action_mask <= 0, MASKED_LOGIT)


class QNetwork(nn.Module):
    """``Q(s, ·)`` over all discrete actions; masked actions get a large negative value."""

    def __init__(self, obs_dim: int, num_actions: int, hidden: list[int]):
        super().__init__()
        self.network = build_mlp(obs_dim, hidden, num_actions)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        return masked_logits(self.network(obs), action_mask)


class CategoricalActor(nn.Module):
    """Masked categorical policy ``pi(a | s)``."""

    def __init__(self, obs_dim: int, num_actions: int, hidden: list[int]):
        super().__init__()
        self.network = build_mlp(obs_dim, hidden, num_actions)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        return masked_logits(self.network(obs), action_mask)

    def log_prob_and_entropy(
        self, obs: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = Categorical(logits=self(obs, action_mask))
        return distribution.log_prob(actions), distribution.entropy()


class _TorchPolicy(Policy):
    """Greedy deployment; softmax(temperature) when a stochastic policy is requested."""

    def __init__(self, module: nn.Module, *, temperature: float = 1.0):
        super().__init__()
        self.module = module.eval()
        self.device = next(module.parameters()).device
        self.temperature = float(temperature)

    def scores(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def action_probabilities(self, obs, info, deterministic: bool = True) -> np.ndarray:
        obs_tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(np.asarray(info["action_mask"], dtype=np.float32), device=self.device).unsqueeze(
            0
        )
        with torch.no_grad():
            scores = self.scores(obs_tensor, mask_tensor).squeeze(0)
            probabilities = torch.softmax(scores / max(self.temperature, 1e-6), dim=-1).cpu().numpy()
        return greedy(probabilities) if deterministic else probabilities


class QPolicy(_TorchPolicy):
    """Acts greedily on one Q-network, or on the minimum of several (twin critics)."""

    def __init__(self, *q_networks: nn.Module, temperature: float = 1.0):
        super().__init__(nn.ModuleList(q_networks), temperature=temperature)

    def scores(self, obs, mask):
        return torch.stack([q(obs, mask) for q in self.module], dim=0).min(dim=0).values


class ActorPolicy(_TorchPolicy):
    def scores(self, obs, mask):
        return self.module(obs, mask)
