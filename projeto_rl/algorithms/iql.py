"""Discrete Implicit Q-Learning (Kostrikov et al., 2022)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from projeto_rl.algorithms import register_algorithm
from projeto_rl.algorithms.base import OfflineAlgorithm, td_target
from projeto_rl.algorithms.networks import ActorPolicy, CategoricalActor, QNetwork, build_mlp, hidden_dims


class IQLNetworks(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: list[int]):
        super().__init__()
        self.actor = CategoricalActor(obs_dim, num_actions, hidden)
        self.q1 = QNetwork(obs_dim, num_actions, hidden)
        self.q2 = QNetwork(obs_dim, num_actions, hidden)
        self.value = build_mlp(obs_dim, hidden, 1)


@register_algorithm
class IQL(OfflineAlgorithm):
    name = "iql"
    method_id = "iql"
    description = "Discrete IQL: expectile value, twin critics, advantage-weighted actor."
    validation_objective = "validation IQL critic, value, and actor objective"

    def build(self):
        return IQLNetworks(self.spec.obs_dim, self.spec.num_actions, hidden_dims(self.config))

    def loss(self, batch):
        cfg, net = self.config, self.model
        q1 = net.q1(batch.obs, batch.action_mask).gather(1, batch.action.unsqueeze(1)).squeeze(1)
        q2 = net.q2(batch.obs, batch.action_mask).gather(1, batch.action.unsqueeze(1)).squeeze(1)
        q_min = torch.minimum(q1, q2)
        with torch.no_grad():
            target = td_target(batch, net.value(batch.next_obs).view(-1), cfg["gamma"])
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        value = net.value(batch.obs).view(-1)
        error = q_min.detach() - value
        expectile = float(cfg.get("iql_expectile_tau", 0.7))
        value_loss = (torch.where(error < 0.0, expectile, 1.0 - expectile) * error.pow(2)).mean()

        weights = torch.exp(float(cfg.get("iql_temperature_beta", 3.0)) * (q_min.detach() - value.detach()))
        weights = weights.clamp(max=float(cfg.get("iql_advantage_clip", 100.0)))
        log_prob, entropy = net.actor.log_prob_and_entropy(batch.obs, batch.action, batch.action_mask)
        actor_loss = -(weights * log_prob).mean()

        loss = (
            float(cfg.get("iql_q_loss_weight", 1.0)) * q_loss
            + float(cfg.get("iql_value_loss_weight", 1.0)) * value_loss
            + float(cfg.get("iql_actor_loss_weight", 1.0)) * actor_loss
        )
        return loss, {
            "q_loss": float(q_loss.item()),
            "value_loss": float(value_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "entropy": float(entropy.mean().item()),
            "advantage_weight": float(weights.mean().item()),
        }

    def make_policy(self):
        return ActorPolicy(self.model.actor, temperature=float(self.config.get("policy_temperature", 1.0)))
