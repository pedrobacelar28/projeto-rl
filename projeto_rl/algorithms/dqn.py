"""Offline DQN (fitted Q-iteration with a soft-updated target network)."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from projeto_rl.algorithms import register_algorithm
from projeto_rl.algorithms.base import OfflineAlgorithm, td_target
from projeto_rl.algorithms.networks import QNetwork, QPolicy, hidden_dims


@register_algorithm
class DQN(OfflineAlgorithm):
    name = "dqn"
    method_id = "dqn"
    description = "Offline DQN: masked one-step TD regression on logged transitions."
    validation_objective = "validation TD loss"

    def build(self):
        network = QNetwork(self.spec.obs_dim, self.spec.num_actions, hidden_dims(self.config))
        self.target = copy.deepcopy(network).to(self.device)
        return network

    def loss(self, batch):
        with torch.no_grad():
            next_value = self.target(batch.next_obs, batch.next_action_mask).max(dim=1).values
            target = td_target(batch, next_value, self.config["gamma"])
        q_value = self.model(batch.obs).gather(1, batch.action.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q_value, target)
        return loss, {"td_loss": float(loss.item())}

    @torch.no_grad()
    def after_update(self, step: int) -> None:
        if step % int(self.config["target_network_frequency"]) == 0:
            tau = float(self.config["tau"])
            for target, source in zip(self.target.parameters(), self.model.parameters()):
                target.data.copy_(tau * source.data + (1.0 - tau) * target.data)

    def make_policy(self):
        return QPolicy(self.model, temperature=float(self.config.get("policy_temperature", 1.0)))
