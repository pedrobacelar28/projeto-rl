"""T-JEPA: a transformer over column tokens pretrained to predict the latent of masked columns."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from projeto_rl.algorithms.base import cpu_state_dict
from projeto_rl.envs.base import CODED_KINDS
from projeto_rl.representations import register_representation
from projeto_rl.representations.base import Representation
from projeto_rl.representations.spr import ema_update
from projeto_rl.utils.torch_utils import get_torch_device

# Standard deviation of the truncated-normal initialization of the released code.
INIT_STD = 0.02
# Rows encoded at once by ``transform``; it bounds memory and changes no value.
TRANSFORM_CHUNK = 16384


def tjepa_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error over every predicted token entry (Eq. 5 of the paper, up to a constant factor)."""

    return F.mse_loss(prediction, target)


def sincos_positions(count: int, dim: int) -> torch.Tensor:
    """Fixed sine-cosine embedding of positions ``0..count-1``: ``[sin(p w_k), cos(p w_k)]``, ``w_k = 10000^(-2k/dim)``."""

    omega = 1.0 / 10000.0 ** (torch.arange(dim // 2, dtype=torch.float64) / (dim / 2.0))
    angles = torch.arange(count, dtype=torch.float64)[:, None] * omega[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=1).float()


def mask_sizes(num_features: int, min_share: float, max_share: float) -> tuple[int, int]:
    """Smallest and largest number of columns in one mask: the shares of the table, rounded."""

    return round(num_features * min_share), round(num_features * max_share)


def sample_masks(
    rng: np.random.Generator, batch_size: int, num_features: int, context_size: int, target_size: int, num_targets: int
) -> tuple[np.ndarray, np.ndarray]:
    """Column indices per row: ``context [B, context_size]`` and ``targets [num_targets, B, target_size]``.

    Targets are drawn among the columns outside the row's context, so a target
    never overlaps the context; two targets of the same row may overlap.
    """

    order = rng.permuted(np.tile(np.arange(num_features), (batch_size, 1)), axis=1)
    context, rest = order[:, :context_size], order[:, context_size:]
    targets = np.stack([rng.permuted(rest, axis=1)[:, :target_size] for _ in range(num_targets)])
    return context, targets


def _take(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``tokens[b, index[b, i]]`` for every row ``b``: ``[B, N, h]`` and ``[B, n]`` to ``[B, n, h]``."""

    return tokens.gather(1, index[:, :, None].expand(-1, -1, tokens.shape[-1]))


def _transformer(dim: int, num_layers: int, num_heads: int, feedforward_dim: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(dim, num_heads, feedforward_dim, dropout=0.0, batch_first=True)
    return nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)


def _reset_parameters(module: nn.Module, generator: torch.Generator) -> None:
    """The released code's initialization, drawn from an explicit generator.

    Weights, embeddings, and tokens are truncated normal; biases are zero;
    layer normalization keeps its construction values (weight 1, bias 0).
    """

    normalization = {id(p) for layer in module.modules() if isinstance(layer, nn.LayerNorm) for p in layer.parameters()}
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if id(parameter) in normalization:
                continue
            if name.endswith("bias"):
                parameter.zero_()
            else:
                nn.init.trunc_normal_(parameter, std=INIT_STD, generator=generator)


class Tokenizer(nn.Module):
    """One learned layer per column and the [REG] token.

    A scalar column goes through its own ``Linear(1, h)``. A coded column
    goes through its own ``Linear(cardinality, h)`` on the one-hot code, which
    is the embedding row of that code.
    """

    def __init__(self, features, hidden_dim: int):
        super().__init__()
        self.columns = nn.ModuleList(
            nn.Embedding(int(feature.cardinality), hidden_dim)
            if feature.kind in CODED_KINDS
            else nn.Linear(1, hidden_dim)
            for feature in features
        )
        self.reg_token = nn.Parameter(torch.empty(hidden_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        tokens = [
            column(obs[:, index].long()) if isinstance(column, nn.Embedding) else column(obs[:, index : index + 1])
            for index, column in enumerate(self.columns)
        ]
        return torch.stack(tokens, dim=1)


class Encoder(nn.Module):
    """Column tokens (all, or the ``keep`` ones) plus [REG] through a transformer; [REG] is dropped from the output."""

    def __init__(self, features, hidden_dim: int, num_layers: int, num_heads: int, feedforward_dim: int):
        super().__init__()
        self.tokenizer = Tokenizer(features, hidden_dim)
        self.transformer = _transformer(hidden_dim, num_layers, num_heads, feedforward_dim)

    def forward(self, obs: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.tokenizer(obs)
        if keep is not None:
            tokens = _take(tokens, keep)
        reg = self.tokenizer.reg_token.expand(len(tokens), 1, -1)
        return self.transformer(torch.cat([tokens, reg], dim=1))[:, :-1]


class Predictor(nn.Module):
    """Predicts the target encoder's tokens of the target columns from the context tokens."""

    def __init__(
        self, num_features: int, hidden_dim: int, dim: int, num_layers: int, num_heads: int, feedforward_dim: int
    ):
        super().__init__()
        self.embed = nn.Linear(hidden_dim, dim)
        self.register_buffer("positions", sincos_positions(num_features, dim))
        self.mask_token = nn.Parameter(torch.empty(dim))
        self.transformer = _transformer(dim, num_layers, num_heads, feedforward_dim)
        self.norm = nn.LayerNorm(dim)
        self.project = nn.Linear(dim, hidden_dim)

    def forward(self, context: torch.Tensor, context_index: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
        visible = self.embed(context) + self.positions[context_index]
        queries = self.mask_token + self.positions[target_index]
        output = self.transformer(torch.cat([visible, queries], dim=1))[:, context.shape[1] :]
        return self.project(self.norm(output))


@register_representation
class TJEPA(Representation):
    """T-JEPA, used as frozen pretraining of a per-row encoder.

    Thimonier, De Melo Costa, Popineau, Rimmel, and Doan, "T-JEPA:
    Augmentation-Free Self-Supervised Learning for Tabular Data", ICLR 2025
    (arXiv:2410.05016). Released code: https://github.com/jose-melo/t-jepa
    (MIT), read at commit 98dda99; nothing is copied from it.

    Kept from the paper (Sec. 3 and Appendix H): numeric columns are
    standardized and coded columns are one-hot; every column has its own
    linear layer into a token; a context encoder sees a subset of the columns
    of a row, a target encoder sees all of them, and both also see a learned
    [REG] token that is never masked, never predicted, and dropped from every
    output; the target encoder is an exponential moving average of the
    context encoder; a narrower transformer predicts the target encoder's
    tokens of the masked columns from the context tokens, a learned mask token,
    and a positional embedding; the loss is the squared distance between the
    two (Eq. 5). Four target masks per row, one [REG] token, a batch of 512,
    a learning rate annealed to zero by a cosine, and a moving-average
    coefficient that grows from 0.996 to 1 are the paper's settings
    (Appendix A.2); the representation comes from the context encoder
    (Sec. 4.1).

    The paper leaves these to its code, which is followed: the masks hold a
    share of the columns drawn per batch and their defaults are 20% to 40%;
    positions are a fixed sine-cosine embedding; the weights start from a
    truncated normal of standard deviation 0.02; the transformer sizes are the
    code's defaults, all inside the paper's search space (Table 5).

    Adaptations:

    - The paper trains for a number of epochs, with model selection by a
      linear probe on labels. Here there are no labels: the encoder trains
      for ``tjepa_train_steps`` steps and the last one is kept.
    - The paper's downstream models learn a projection of the ``d x h`` token
      matrix together with the task. The encoder is frozen here, so the
      matrix is flattened to ``d * h`` values; the first layer of every
      downstream network is then the paper's "linear flatten" projection.
    - ``ordinal`` and ``cyclic`` columns are one-hot, like ``categorical``
      ones; ``binary`` columns are scalars and are not standardized.
    - The index and feature-type embeddings of the paper are added to a
      per-column bias, so they are the bias of each column's layer (the row of
      each code, for coded columns) rather than separate parameters.
    - No dropout (the paper searches it in ``[0, 0.01]``) and Adam, which is
      the paper's AdamW with the weight decay of zero of the code's launch
      script; no learning-rate warm-up.
    - The released code differs from the paper in three places, and the paper
      is followed: the code rescales numeric columns to ``[0, 1]``, lets its
      mask indices address the [REG] token, and exports the target encoder
      with the [REG] token.

    The fitted instance keeps its encoder on the training device, where it
    transforms the splits in bulk; an instance rebuilt from a checkpoint runs
    on the CPU, where evaluation transforms one observation per simulator step.
    """

    name = "tjepa"

    def __init__(self, features, config: dict[str, Any]):
        super().__init__(features, config)
        self.hidden_dim = int(self._setting("tjepa_hidden_dim"))
        self.architecture = (
            self.hidden_dim,
            int(self._setting("tjepa_num_layers")),
            int(self._setting("tjepa_num_heads")),
            int(self._setting("tjepa_feedforward_dim")),
        )
        self.predictor_architecture = (
            int(self._setting("tjepa_predictor_dim")),
            int(self._setting("tjepa_predictor_layers")),
            int(self._setting("tjepa_predictor_heads")),
            int(self._setting("tjepa_feedforward_dim")),
        )
        if min(self.architecture + self.predictor_architecture) <= 0:
            raise ValueError("Every tjepa_* size (dimensions, layers, heads) must be > 0")
        if self.predictor_architecture[0] % 2:
            raise ValueError("tjepa_predictor_dim must be even (sine-cosine positions)")
        self.context_sizes = self._mask_sizes("context")
        self.target_sizes = self._mask_sizes("target")
        if self.context_sizes[1] + self.target_sizes[1] > self.input_dim:
            raise ValueError(
                f"The largest tjepa context ({self.context_sizes[1]}) and target ({self.target_sizes[1]}) masks "
                f"do not fit together in {self.input_dim} columns"
            )
        self.numeric = np.array([feature.kind == "numeric" for feature in self.features])

    def _setting(self, key: str):
        if key not in self.config:
            raise ValueError(f"The tjepa representation needs {key!r} in the run config (configs/protocol.json)")
        return self.config[key]

    def _mask_sizes(self, role: str) -> tuple[int, int]:
        low, high = mask_sizes(
            self.input_dim,
            float(self._setting(f"tjepa_min_{role}_share")),
            float(self._setting(f"tjepa_max_{role}_share")),
        )
        if not 1 <= low <= high:
            raise ValueError(
                f"tjepa_min_{role}_share and tjepa_max_{role}_share give masks of {low} to {high} columns "
                f"on a table of {self.input_dim}; they must hold at least one column"
            )
        return low, high

    def _build_encoder(self) -> Encoder:
        # Construction draws PyTorch's default initialization from the global
        # stream; the fork discards those draws. The weights then come from the
        # fit's own generator or from a checkpoint.
        with torch.random.fork_rng(devices=[]):
            return Encoder(self.features, *self.architecture)

    def fit(self, train, *, seed: int) -> None:
        steps, batch_size = int(self._setting("tjepa_train_steps")), int(self._setting("tjepa_batch_size"))
        learning_rate, num_targets = (
            float(self._setting("tjepa_learning_rate")),
            int(self._setting("tjepa_target_masks")),
        )
        ema_start, ema_end = float(self._setting("tjepa_ema_start")), float(self._setting("tjepa_ema_end"))
        if steps <= 0 or batch_size <= 0 or learning_rate <= 0.0 or num_targets <= 0:
            raise ValueError(
                "tjepa_train_steps, tjepa_batch_size, tjepa_learning_rate, and tjepa_target_masks must be > 0"
            )
        if not 0.0 <= ema_start <= ema_end <= 1.0:
            raise ValueError("tjepa_ema_start and tjepa_ema_end must satisfy 0 <= start <= end <= 1")
        raw = np.asarray(train.observations, dtype=np.float64)
        mean = np.where(self.numeric, raw.mean(axis=0), 0.0)
        std = np.where(self.numeric, raw.std(axis=0), 1.0)
        constant = [feature.name for feature, value in zip(self.features, std, strict=True) if value == 0.0]
        if constant:
            raise ValueError(f"Cannot standardize constant train columns: {constant}")
        device = get_torch_device()
        inputs = torch.as_tensor((raw - mean) / std, dtype=torch.float32, device=device)

        encoder = self._build_encoder()
        with torch.random.fork_rng(devices=[]):
            predictor = Predictor(self.input_dim, self.hidden_dim, *self.predictor_architecture)
        online = nn.ModuleList([encoder, predictor])
        _reset_parameters(online, torch.Generator().manual_seed(seed))
        target_encoder = copy.deepcopy(encoder).requires_grad_(False)
        for module in (online, target_encoder):
            module.to(device)
        optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        rng = np.random.default_rng(seed)

        #: Training loss of every gradient step, kept for diagnostics.
        self.loss_history: list[float] = []
        for step in range(steps):
            batch = inputs[torch.as_tensor(rng.integers(0, len(inputs), size=batch_size), device=device)]
            context_size = int(rng.integers(self.context_sizes[0], self.context_sizes[1] + 1))
            target_size = int(rng.integers(self.target_sizes[0], self.target_sizes[1] + 1))
            context_index, target_index = (
                torch.as_tensor(index, device=device)
                for index in sample_masks(rng, batch_size, self.input_dim, context_size, target_size, num_targets)
            )
            # Every target mask is predicted from the same context, so the
            # context is repeated once per mask: [num_targets * B, ...].
            target_index = target_index.flatten(0, 1)
            with torch.no_grad():
                target = _take(target_encoder(batch).repeat(num_targets, 1, 1), target_index)
            context = encoder(batch, keep=context_index)
            prediction = predictor(
                context.repeat(num_targets, 1, 1), context_index.repeat(num_targets, 1), target_index
            )
            loss = tjepa_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            ema_update(target_encoder, encoder, ema_start + (ema_end - ema_start) * step / steps)
            self.loss_history.append(float(loss.item()))

        self.encoder, self.mean, self.std = encoder.eval(), mean, std
        latent = self._transform(np.asarray(train.observations, dtype=np.float32))
        if not np.isfinite(latent).all():
            raise ValueError("T-JEPA diverged: the encoder outputs non-finite values on the train split")
        if not latent.std(axis=0).any():
            raise ValueError("T-JEPA collapsed: the representation is the same for every row of the train split")

    def _transform(self, obs):
        device = next(self.encoder.parameters()).device
        scaled = torch.as_tensor((obs - self.mean) / self.std, dtype=torch.float32)
        with torch.no_grad():
            chunks = [self.encoder(chunk.to(device)).flatten(1).cpu() for chunk in scaled.split(TRANSFORM_CHUNK)]
        return torch.cat(chunks).numpy()

    @property
    def output_dim(self) -> int:
        return self.input_dim * self.hidden_dim

    def fitted_state(self):
        return {"encoder": cpu_state_dict(self.encoder), "mean": self.mean.tolist(), "std": self.std.tolist()}

    def load_fitted_state(self, state) -> None:
        self.encoder = self._build_encoder().eval()
        self.encoder.load_state_dict(state["encoder"])
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.std = np.asarray(state["std"], dtype=np.float64)
