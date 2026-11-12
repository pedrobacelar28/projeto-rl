"""SPR: a latent encoder pretrained to predict its own next latent from the logged transitions."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from projeto_rl.algorithms.base import cpu_state_dict
from projeto_rl.algorithms.networks import build_mlp
from projeto_rl.representations import register_representation
from projeto_rl.representations.base import Representation
from projeto_rl.utils.torch_utils import get_torch_device


def spr_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative cosine similarity averaged over the batch (Eq. 4 of the paper, one step)."""

    return -F.cosine_similarity(prediction, target, dim=-1).mean()


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """``theta_m <- tau * theta_m + (1 - tau) * theta_o`` (Eq. 2 of the paper)."""

    for target_parameter, online_parameter in zip(target.parameters(), online.parameters(), strict=True):
        target_parameter.mul_(tau).add_(online_parameter, alpha=1.0 - tau)


def _reset_parameters(module: nn.Module, generator: torch.Generator) -> None:
    """PyTorch's default ``nn.Linear`` initialization, drawn from an explicit generator."""

    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            bound = layer.in_features**-0.5
            with torch.no_grad():
                layer.weight.uniform_(-bound, bound, generator=generator)
                layer.bias.uniform_(-bound, bound, generator=generator)


class Encoder(nn.Module):
    """MLP from raw observations to the latent; hidden units are dropped only when a generator is given."""

    def __init__(self, input_dim: int, hidden: list[int], latent_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in pairwise([input_dim, *hidden, latent_dim]))
        self.dropout = float(dropout)

    def forward(self, obs: torch.Tensor, dropout_generator: torch.Generator | None = None) -> torch.Tensor:
        hidden = obs
        for layer in self.layers[:-1]:
            hidden = torch.relu(layer(hidden))
            if dropout_generator is not None and self.dropout > 0.0:
                keep = 1.0 - self.dropout
                hidden = hidden * torch.bernoulli(torch.full_like(hidden, keep), generator=dropout_generator) / keep
        return self.layers[-1](hidden)


@register_representation
class SPR(Representation):
    """Self-Predictive Representations, used as frozen pretraining.

    Schwarzer, Anand, Goel, Hjelm, Courville, and Bachman, "Data-Efficient
    Reinforcement Learning with Self-Predictive Representations", ICLR 2021
    (arXiv:2007.05929).

    Kept from the paper: an online encoder, an action-conditioned transition
    model in latent space, an online projection and a linear prediction head,
    target encoder and projection that are exponential moving averages of the
    online ones (Eq. 2), and the negative cosine similarity between the
    predicted and the target projection of the next latent (Eq. 4). The
    target coefficient ``tau = 0.99`` and the dropout of 0.5 in both encoders
    are the paper's settings for training without data augmentation, which is
    the case here (Sec. 2.4 and 2.5).

    Adaptations:

    - The paper trains the encoder jointly with Rainbow
      (``L_RL + lambda * L_SPR``). Here the SPR loss is the only objective and
      the encoder is frozen afterwards, as every representation in this
      project is.
    - One prediction step (the logged transition) instead of ``K = 5``.
    - Observations are table rows, so the convolutional encoder and
      transition model become MLPs (no batch normalization), with the one-hot
      action concatenated to the latent. Dropout follows each hidden layer.
    - The paper's projection is the first layer of the Q-learning head; there
      is no such head here, so it is a linear layer of its own.
    - The encoder reads the raw observation, with no input scaling. Its
      latent is z-scored per dimension with train-split statistics, so the
      downstream networks see inputs of unit scale whatever the scale the
      encoder settles on.

    The deployed encoder runs on the CPU: evaluation transforms one
    observation per simulator step.
    """

    name = "spr"

    def __init__(self, features, config: dict[str, Any]):
        super().__init__(features, config)
        self.latent_dim = int(self._setting("spr_latent_dim"))
        self.hidden = [int(self._setting("spr_hidden_dim"))] * int(self._setting("spr_hidden_layers"))
        self.dropout = float(self._setting("spr_dropout"))
        if self.latent_dim <= 0 or not self.hidden or self.hidden[0] <= 0:
            raise ValueError("spr_latent_dim, spr_hidden_dim, and spr_hidden_layers must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("spr_dropout must be in [0, 1)")

    def _setting(self, key: str):
        if key not in self.config:
            raise ValueError(f"The spr representation needs {key!r} in the run config (configs/protocol.json)")
        return self.config[key]

    def _build_encoder(self) -> Encoder:
        return Encoder(self.input_dim, self.hidden, self.latent_dim, self.dropout)

    def fit(self, train, *, seed: int) -> None:
        steps, batch_size = int(self._setting("spr_train_steps")), int(self._setting("spr_batch_size"))
        learning_rate, tau = float(self._setting("spr_learning_rate")), float(self._setting("spr_target_tau"))
        if steps <= 0 or batch_size <= 0 or learning_rate <= 0.0:
            raise ValueError("spr_train_steps, spr_batch_size, and spr_learning_rate must be > 0")
        if not 0.0 <= tau < 1.0:
            raise ValueError("spr_target_tau must be in [0, 1)")
        device = get_torch_device()
        num_actions = int(train.num_actions)
        obs = torch.as_tensor(train.observations, dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(train.next_observations, dtype=torch.float32, device=device)
        actions = F.one_hot(torch.as_tensor(train.data["actions"], dtype=torch.int64), num_actions).float().to(device)

        # Construction draws PyTorch's default initialization from the global
        # stream; the fork discards those draws, and every weight is redrawn
        # from a generator owned by this fit.
        with torch.random.fork_rng(devices=[]):
            encoder, target_encoder = self._build_encoder(), self._build_encoder()
            transition = build_mlp(self.latent_dim + num_actions, self.hidden[:1], self.latent_dim)
            projection = nn.Linear(self.latent_dim, self.latent_dim)
            target_projection = nn.Linear(self.latent_dim, self.latent_dim)
            predictor = nn.Linear(self.latent_dim, self.latent_dim)
        online = nn.ModuleList([encoder, transition, projection, predictor])
        _reset_parameters(online, torch.Generator().manual_seed(seed))
        target_encoder.load_state_dict(encoder.state_dict())
        target_projection.load_state_dict(projection.state_dict())
        for module in (online, target_encoder, target_projection):
            module.to(device)
        optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate)
        batch_rng = np.random.default_rng(seed)
        dropout_generator = torch.Generator(device=device).manual_seed(seed)

        #: Training loss of every gradient step, kept for diagnostics.
        self.loss_history: list[float] = []
        for _ in range(steps):
            index = torch.as_tensor(batch_rng.integers(0, len(obs), size=batch_size), device=device)
            latent = encoder(obs[index], dropout_generator)
            predicted_next = transition(torch.cat([latent, actions[index]], dim=-1))
            with torch.no_grad():
                target = target_projection(target_encoder(next_obs[index], dropout_generator))
            loss = spr_loss(predictor(projection(predicted_next)), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_update(target_encoder, encoder, tau)
            ema_update(target_projection, projection, tau)
            self.loss_history.append(float(loss.item()))

        self.encoder = encoder.cpu()
        latent = self._latent(np.asarray(train.observations, dtype=np.float32))
        if not np.isfinite(latent).all():
            raise ValueError("SPR diverged: the encoder outputs non-finite latents on the train split")
        mean, std = latent.mean(axis=0), latent.std(axis=0)
        collapsed = np.flatnonzero(std == 0.0).tolist()
        if collapsed:
            raise ValueError(f"SPR collapsed: latent dimensions {collapsed} are constant on the train split")
        self.latent_mean, self.latent_std = mean, std

    def _latent(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            latent = self.encoder(torch.as_tensor(obs))
        return latent.numpy().astype(np.float64)

    def _transform(self, obs):
        return (self._latent(obs) - self.latent_mean) / self.latent_std

    @property
    def output_dim(self) -> int:
        return self.latent_dim

    def fitted_state(self):
        return {
            "encoder": cpu_state_dict(self.encoder),
            "latent_mean": self.latent_mean.tolist(),
            "latent_std": self.latent_std.tolist(),
        }

    def load_fitted_state(self, state) -> None:
        self.encoder = self._build_encoder()
        self.encoder.load_state_dict(state["encoder"])
        self.latent_mean = np.asarray(state["latent_mean"], dtype=np.float64)
        self.latent_std = np.asarray(state["latent_std"], dtype=np.float64)
