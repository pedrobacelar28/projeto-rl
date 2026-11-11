"""TD-JEPA: a state encoder pretrained to predict, by temporal differences, the latents its policies will visit."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from projeto_rl.algorithms.base import cpu_state_dict
from projeto_rl.algorithms.networks import masked_logits
from projeto_rl.representations import register_representation
from projeto_rl.representations.base import Representation
from projeto_rl.representations.spr import ema_update
from projeto_rl.representations.standardize import Standardize
from projeto_rl.utils.torch_utils import get_torch_device

# Networks in each predictor; targets and action choices read their mean (Appendix E.4).
TWINS = 2
# Gain of the orthogonal initialization of the predictors in the released code (the one for ReLU).
PREDICTOR_GAIN = 2.0**0.5


def to_sphere(vectors: torch.Tensor) -> torch.Tensor:
    """Each row rescaled to norm ``sqrt(d)``, so its entries have unit mean square."""

    return vectors.shape[-1] ** 0.5 * F.normalize(vectors, dim=-1)


def td_jepa_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared distance to the target, summed over dimensions and averaged over twins and rows."""

    return (prediction - target).pow(2).sum(dim=-1).mean()


def orthonormality_loss(latent: torch.Tensor) -> torch.Tensor:
    """``(1/2) mean_{i != j} (e_i . e_j)^2 - mean_i |e_i|^2`` over the rows of a batch (Algorithm 1)."""

    gram = latent @ latent.T
    rows = len(latent)
    off_diagonal = gram.pow(2).sum() - gram.diagonal().pow(2).sum()
    return 0.5 * off_diagonal / (rows * (rows - 1)) - gram.diagonal().mean()


def sample_tasks(goals: torch.Tensor, goal_share: float, generator: torch.Generator) -> torch.Tensor:
    """One task vector ``z`` per row, on the sphere of radius ``sqrt(d)``.

    With probability ``goal_share`` it is the task latent of another row of
    the batch (``goals``, shuffled); otherwise it is uniform on the sphere.
    """

    device = goals.device
    uniform = to_sphere(torch.randn(goals.shape, generator=generator, device=device))
    shuffled = to_sphere(goals[torch.randperm(len(goals), generator=generator, device=device)])
    from_goal = torch.rand((len(goals), 1), generator=generator, device=device) < goal_share
    return torch.where(from_goal, shuffled, uniform)


def greedy_actions(predictions: torch.Tensor, z: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """``argmax_a T(latent, a, z) . z`` over the valid actions; ``predictions`` is ``[TWINS, B, A, d]``."""

    scores = (predictions.mean(dim=0) * z[:, None, :]).sum(dim=-1)
    return masked_logits(scores, action_mask).argmax(dim=1)


def _embedding(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
    )


class Encoder(nn.Module):
    """MLP from the scaled observation to a latent on the sphere of radius ``sqrt(d)``."""

    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int, output_dim: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.network = nn.Sequential(*layers, nn.Linear(hidden_dim, output_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return to_sphere(self.network(obs))


class Predictor(nn.Module):
    """Twin networks from ``(latent, z, one-hot action)`` to the other encoder's space: ``[TWINS, B, output_dim]``."""

    def __init__(self, latent_dim: int, task_dim: int, num_actions: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.num_actions = num_actions
        self.task_embeddings = nn.ModuleList(_embedding(latent_dim + task_dim, hidden_dim) for _ in range(TWINS))
        self.action_embeddings = nn.ModuleList(_embedding(latent_dim + num_actions, hidden_dim) for _ in range(TWINS))
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
            for _ in range(TWINS)
        )

    def _embed_task(self, latent: torch.Tensor, z: torch.Tensor) -> list[torch.Tensor]:
        with_task = torch.cat([latent, z], dim=-1)
        return [embed(with_task) for embed in self.task_embeddings]

    def _predict(self, latent: torch.Tensor, action: torch.Tensor, task_embedded: list[torch.Tensor]) -> torch.Tensor:
        with_action = torch.cat([latent, action], dim=-1)
        return torch.stack(
            [
                head(torch.cat([embed(with_action), task], dim=-1))
                for embed, head, task in zip(self.action_embeddings, self.heads, task_embedded, strict=True)
            ]
        )

    def forward(self, latent: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._predict(latent, action, self._embed_task(latent, z))

    def all_actions(self, latent: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """The prediction for every action of every row: ``[TWINS, B, A, output_dim]``."""

        rows = len(latent)
        actions = torch.eye(self.num_actions, device=latent.device).repeat(rows, 1)
        # The task embedding does not depend on the action: it is computed once per row.
        task_embedded = [task.repeat_interleave(self.num_actions, dim=0) for task in self._embed_task(latent, z)]
        prediction = self._predict(latent.repeat_interleave(self.num_actions, dim=0), actions, task_embedded)
        return prediction.unflatten(1, (rows, self.num_actions))


class Networks(nn.Module):
    """The state encoder ``phi``, the task encoder ``psi``, and one predictor reading each of them."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        state_dim: int,
        task_dim: int,
        encoder_hidden_dim: int,
        encoder_hidden_layers: int,
        predictor_hidden_dim: int,
    ):
        super().__init__()
        self.state_encoder = Encoder(input_dim, encoder_hidden_dim, encoder_hidden_layers, state_dim)
        self.task_encoder = Encoder(input_dim, encoder_hidden_dim, encoder_hidden_layers, task_dim)
        self.state_predictor = Predictor(state_dim, task_dim, num_actions, predictor_hidden_dim, task_dim)
        self.task_predictor = Predictor(task_dim, task_dim, num_actions, predictor_hidden_dim, state_dim)


def _reset_parameters(module: nn.Module, generator: torch.Generator, gain: float) -> None:
    """The released code's initialization, drawn from an explicit generator: orthogonal weights, zero biases."""

    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain, generator=generator)
            nn.init.zeros_(layer.bias)


@torch.no_grad()
def td_targets(
    online: Networks,
    target: Networks,
    next_obs: torch.Tensor,
    z: torch.Tensor,
    next_action_mask: torch.Tensor,
    continuing: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Targets of the two predictors (Eq. 9 of the paper and its mirror image).

    ``psi^-(s') + gamma T_phi^-(phi^-(s'), a', z)`` for the state predictor and
    ``phi^-(s') + gamma T_psi^-(psi^-(s'), a', z)`` for the task predictor,
    where ``a'`` is the action of the policy of task ``z`` at ``s'``.
    ``continuing`` is ``[B, 1]``: ``gamma`` on the rows that go on, 0 on the
    terminal ones.

    Choosing ``a'`` evaluates every action of every row and keeps only an
    argmax, so on a GPU that pass runs in half precision.
    """

    next_state, next_task = target.state_encoder(next_obs), target.task_encoder(next_obs)
    with torch.autocast(next_obs.device.type, dtype=torch.bfloat16, enabled=next_obs.is_cuda):
        choice = greedy_actions(online.state_predictor.all_actions(next_state, z), z, next_action_mask)
    next_action = F.one_hot(choice, online.state_predictor.num_actions).float()
    state_target = next_task + continuing * target.state_predictor(next_state, z, next_action).mean(dim=0)
    task_target = next_state + continuing * target.task_predictor(next_task, z, next_action).mean(dim=0)
    return state_target, task_target


@register_representation
class TDJEPA(Representation):
    """TD-JEPA, used as frozen pretraining of a state encoder.

    Bagatella, Pirotta, Touati, Lazaric, and Tirinzoni, "TD-JEPA:
    Latent-predictive Representations for Zero-Shot Reinforcement Learning",
    ICLR 2026 (arXiv:2510.00739). Released code:
    https://github.com/facebookresearch/td_jepa (CC BY-NC 4.0), read for the
    details the paper leaves out; nothing is copied from it.

    The method learns from logged transitions without rewards. A family of
    policies ``pi_z`` is indexed by a task vector ``z``; a state encoder
    ``phi`` and a predictor ``T_phi(phi(s), a, z)`` are trained so that the
    predictor gives the discounted sum of the task latents ``psi(s_t)`` that
    ``pi_z`` visits after ``(s, a)``, its successor features. That sum is
    learned from single transitions through its Bellman equation (Eq. 9), so
    the encoder has to keep what decides the long-term future under many
    policies, not only under the one that logged the data. A task encoder
    ``psi`` and its predictor ``T_psi`` are trained by the same loss with the
    roles of the two encoders swapped.

    Kept from the paper and its code: the two encoders with outputs on the
    sphere of radius ``sqrt(d)``; the two predictors, each a pair of twin
    networks with two embedding layers (the first layer-normalized, with
    tanh) whose mean gives targets and action values; target copies of the
    four networks updated by an exponential moving average; the loss of
    Algorithm 1 with its orthonormality regularizer on both encoders; task
    vectors drawn uniformly on the sphere or, with probability
    ``tdjepa_goal_share``, set to the task latent of another row of the
    batch; orthogonal initialization; Adam with one learning rate. The
    squared error has no factor 1/2, as in the code, whose regularization
    coefficients are the ones the paper reports.

    Every size and coefficient in ``configs/protocol.json`` is the default
    of the released code. They are the values of the paper's proprioceptive
    DeepMind Control setting (Table 4: predictors of width 1024, a task
    encoder with two hidden layers of 256 and 50 outputs, a discount of 0.98,
    a batch of 1024, a learning rate of 1e-4, a regularization coefficient
    of 1, goals for half of the rows) except three, where the paper's
    choice depends on the benchmark: the state latent has 50 dimensions
    (256 in Table 4, 50 in the code's OGBench launch script; the tables here
    have 5 to 8 columns); the state encoder has two hidden layers like the
    task encoder (the paper tunes its depth: a linear map in DeepMind
    Control, four hidden layers in OGBench); the moving-average coefficient
    is 0.995 (the paper's OGBench value; 0.999 in DeepMind Control, which
    would leave the targets a tenth of this training behind).

    Adaptations:

    - Only the state encoder is kept, frozen, as the representation, and
      the downstream networks start from scratch: the reuse the paper
      studies for single-task RL from pixels (Sec. 5.4 and Appendix D.3).
      The task encoder, the predictors, and the zero-shot policies are
      discarded after the fit, and the logged reward is never read.
    - Actions are discrete here. The paper trains a Gaussian actor to
      maximize ``T_phi(phi(s), a, z) . z``; with a finite action set that
      maximum is computed exactly over the valid actions, so there is no
      actor network. As for the paper's actor, the action is chosen with the
      online predictor and evaluated with the target one. The predictors
      read the action one-hot.
    - ``tdjepa_train_steps`` gradient steps instead of the paper's one to
      three million. Temporal-difference targets build up slowly, so the
      predictors are far from converged when the encoder is frozen.
    - The paper adds a behavior-cloning term to the actor on datasets with
      narrow action coverage (its Appendix E.6, defined for continuous
      actions through a flow model). It is not used: the policies here are
      the plain ones of Algorithm 1.
    - The paper feeds observations unscaled. Tabular columns have very
      different scales, so every column is z-scored with train-split
      statistics first, exactly as the ``standardize`` representation does.

    The deployed encoder runs on the CPU.
    """

    name = "tdjepa"

    def __init__(self, features, config: dict[str, Any]):
        super().__init__(features, config)
        self.state_dim = int(self._setting("tdjepa_state_dim"))
        self.encoder_hidden = (
            int(self._setting("tdjepa_encoder_hidden_dim")),
            int(self._setting("tdjepa_encoder_hidden_layers")),
        )
        self.sizes = (
            self.state_dim,
            int(self._setting("tdjepa_task_dim")),
            *self.encoder_hidden,
            int(self._setting("tdjepa_predictor_hidden_dim")),
        )
        if min(self.sizes) <= 0:
            raise ValueError("Every tdjepa_* size (dimensions and layers) must be > 0")
        if self.sizes[-1] % 2:
            raise ValueError("tdjepa_predictor_hidden_dim must be even (two embeddings of half the width)")
        self.scaler = Standardize(features, config)

    def _setting(self, key: str):
        if key not in self.config:
            raise ValueError(f"The tdjepa representation needs {key!r} in the run config (configs/protocol.json)")
        return self.config[key]

    def _build_encoder(self) -> Encoder:
        return Encoder(self.input_dim, *self.encoder_hidden, self.state_dim)

    def fit(self, train, *, seed: int) -> None:
        steps, batch_size = int(self._setting("tdjepa_train_steps")), int(self._setting("tdjepa_batch_size"))
        learning_rate, discount = float(self._setting("tdjepa_learning_rate")), float(self._setting("tdjepa_discount"))
        goal_share, ema = float(self._setting("tdjepa_goal_share")), float(self._setting("tdjepa_ema"))
        coefficient = float(self._setting("tdjepa_orthonormality_coef"))
        if steps <= 0 or batch_size < 2 or learning_rate <= 0.0:
            raise ValueError("tdjepa_train_steps and tdjepa_learning_rate must be > 0, tdjepa_batch_size >= 2")
        if not (0.0 <= discount < 1.0 and 0.0 <= ema < 1.0):
            raise ValueError("tdjepa_discount and tdjepa_ema must be in [0, 1)")
        if not 0.0 <= goal_share <= 1.0 or coefficient < 0.0:
            raise ValueError("tdjepa_goal_share must be in [0, 1] and tdjepa_orthonormality_coef >= 0")
        self.scaler.fit(train, seed=seed)
        device = get_torch_device()
        num_actions = int(train.num_actions)
        obs = torch.as_tensor(self.scaler.transform(train.observations), device=device)
        next_obs = torch.as_tensor(self.scaler.transform(train.next_observations), device=device)
        # Checked here because a bad row only breaks the fit when a batch happens to draw it.
        if not torch.isfinite(next_obs).all():
            raise ValueError("The train split has non-finite values in the next observations")
        actions = F.one_hot(torch.as_tensor(train.data["actions"], dtype=torch.int64), num_actions).float().to(device)
        terminals = torch.as_tensor(train.data["terminals"], dtype=torch.float32, device=device)
        continuing = discount * (1.0 - terminals)[:, None]
        next_masks = torch.as_tensor(train.data["next_action_masks"], dtype=torch.float32, device=device)

        # Construction draws PyTorch's default initialization from the global
        # stream; the fork discards those draws, and every weight is redrawn
        # from a generator owned by this fit.
        with torch.random.fork_rng(devices=[]):
            online = Networks(self.input_dim, num_actions, *self.sizes)
        generator = torch.Generator().manual_seed(seed)
        for module, gain in (
            (online.state_encoder, 1.0),
            (online.task_encoder, 1.0),
            (online.state_predictor, PREDICTOR_GAIN),
            (online.task_predictor, PREDICTOR_GAIN),
        ):
            _reset_parameters(module, generator, gain)
        target = copy.deepcopy(online).requires_grad_(False)
        for module in (online, target):
            module.to(device)
        optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate)
        # Batches and task vectors are drawn on the training device.
        sampler = torch.Generator(device=device).manual_seed(seed)

        #: Latent-prediction loss (the two temporal-difference terms) of every gradient step, kept for diagnostics.
        self.loss_history: list[float] = []
        for _ in range(steps):
            index = torch.randint(len(obs), (batch_size,), generator=sampler, device=device)
            with torch.no_grad():
                z = sample_tasks(online.task_encoder(next_obs[index]), goal_share, sampler)
            state_target, task_target = td_targets(
                online, target, next_obs[index], z, next_masks[index], continuing[index]
            )
            state, task = online.state_encoder(obs[index]), online.task_encoder(obs[index])
            prediction_loss = td_jepa_loss(
                online.state_predictor(state, z, actions[index]), state_target
            ) + td_jepa_loss(online.task_predictor(task, z, actions[index]), task_target)
            loss = prediction_loss + coefficient * (orthonormality_loss(state) + orthonormality_loss(task))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_update(target, online, ema)
            self.loss_history.append(float(prediction_loss.item()))

        self.encoder = online.state_encoder.cpu().eval()
        latent = self._transform(np.asarray(train.observations, dtype=np.float32))
        if not np.isfinite(latent).all():
            raise ValueError("TD-JEPA diverged: the encoder outputs non-finite values on the train split")
        if not latent.std(axis=0).any():
            raise ValueError("TD-JEPA collapsed: the representation is the same for every row of the train split")

    def _transform(self, obs):
        with torch.no_grad():
            return self.encoder(torch.as_tensor(self.scaler.transform(obs))).numpy()

    @property
    def output_dim(self) -> int:
        return self.state_dim

    def fitted_state(self):
        return {"scaler": self.scaler.fitted_state(), "encoder": cpu_state_dict(self.encoder)}

    def load_fitted_state(self, state) -> None:
        self.scaler.load_fitted_state(state["scaler"])
        with torch.random.fork_rng(devices=[]):
            self.encoder = self._build_encoder().eval()
        self.encoder.load_state_dict(state["encoder"])
