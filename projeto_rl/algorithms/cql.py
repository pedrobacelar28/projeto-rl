"""Discrete Conservative Q-Learning (Kumar et al., 2020)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from projeto_rl.algorithms import register_algorithm
from projeto_rl.algorithms.base import OfflineAlgorithm, td_target
from projeto_rl.algorithms.networks import QNetwork, QPolicy, hidden_dims


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: list[int]):
        super().__init__()
        self.q1 = QNetwork(obs_dim, num_actions, hidden)
        self.q2 = QNetwork(obs_dim, num_actions, hidden)


@register_algorithm
class CQL(OfflineAlgorithm):
    name = "cql"
    method_id = "cql"
    description = "Discrete CQL: TD regression plus a conservative logsumexp penalty."
    validation_objective = "validation conservative Q objective"

    def build(self):
        return TwinQ(self.spec.obs_dim, self.spec.num_actions, hidden_dims(self.config))

    def loss(self, batch):
        net = self.model
        q1_all = net.q1(batch.obs, batch.action_mask)
        q2_all = net.q2(batch.obs, batch.action_mask)
        q1 = q1_all.gather(1, batch.action.unsqueeze(1)).squeeze(1)
        q2 = q2_all.gather(1, batch.action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = torch.minimum(
                net.q1(batch.next_obs, batch.next_action_mask), net.q2(batch.next_obs, batch.next_action_mask)
            )
            target = td_target(batch, next_q.max(dim=1).values, self.config["gamma"])
        td_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        # Inadmissible actions carry a -1e9 Q-value, so logsumexp runs exactly
        # over the admissible set.
        conservative = 0.5 * (
            torch.logsumexp(q1_all, dim=1).mean() - q1.mean() + torch.logsumexp(q2_all, dim=1).mean() - q2.mean()
        )
        alpha = float(self.config.get("cql_alpha", 1.0))
        loss = td_loss + alpha * conservative
        return loss, {
            "q_loss": float(td_loss.item()),
            "cql_loss": float(conservative.item()),
            "conservative_loss": float((alpha * conservative).item()),
        }

    def make_policy(self):
        return QPolicy(self.model.q1, self.model.q2, temperature=float(self.config.get("policy_temperature", 1.0)))
