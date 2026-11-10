"""The single offline training loop shared by every algorithm.

The run's representation and reward scale are fitted on the train split
only, frozen, and applied to both splits before the algorithm sees any
batch. Model selection is purely offline: the validation split's loss
decides the best checkpoint and early stopping. The simulator is never consulted during
training; it is used only by the final evaluation in :mod:`projeto_rl.runner`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from projeto_rl.algorithms import DataSpec, get_algorithm
from projeto_rl.data.dataset import OfflineTransitionDataset
from projeto_rl.representations import build_representation
from projeto_rl.utils.progress import progress_write, tqdm
from projeto_rl.utils.queue_progress import update_progress
from projeto_rl.utils.seeding import set_seed
from projeto_rl.utils.torch_utils import get_torch_device


@dataclass
class TrainResult:
    history: list[dict[str, float]]
    summary: dict[str, Any]
    best_checkpoint: dict[str, Any]
    last_checkpoint: dict[str, Any]


@dataclass
class _EarlyStopping:
    patience: int
    min_delta: float
    best_loss: float = float("inf")
    best_step: int = -1
    best_sampled: int = -1
    best_checkpoint: dict[str, Any] | None = None
    checks: int = 0
    checks_without_improvement: int = 0
    stopped_at: int | None = None

    def check(self, loss: float, *, step: int, sampled: int, snapshot) -> tuple[bool, bool]:
        """Record one validation check; return ``(improved, should_stop)``."""

        self.checks += 1
        if loss < self.best_loss - self.min_delta or self.best_checkpoint is None:
            self.best_loss, self.best_step, self.best_sampled = float(loss), int(step), int(sampled)
            self.best_checkpoint = snapshot()
            self.checks_without_improvement = 0
            return True, False
        self.checks_without_improvement += 1
        if self.patience > 0 and self.checks_without_improvement >= self.patience:
            self.stopped_at = int(step)
            return False, True
        return False, False


def _format(value: Any) -> str:
    return str(value) if isinstance(value, int) else f"{float(value):.4f}"


def load_splits(config: dict[str, Any]) -> tuple[OfflineTransitionDataset, OfflineTransitionDataset]:
    seed = int(config["seed"])
    train = OfflineTransitionDataset(config["dataset_path"], seed=seed)
    validation = OfflineTransitionDataset(config["val_dataset_path"], seed=seed + 100_000)
    return train, validation


def train_offline(config: dict[str, Any]) -> TrainResult:
    """Optimize ``config['algo']`` on the logged train split for a fixed step budget."""

    total_steps = int(config["offline_train_steps"])
    log_interval = int(config["offline_log_interval"])
    if total_steps <= 0 or log_interval <= 0 or int(config["offline_batch_size"]) <= 0:
        raise ValueError("offline_train_steps, offline_log_interval, and offline_batch_size must be > 0")
    show_progress = bool(config.get("show_progress", True))
    set_seed(int(config["seed"]))
    device = get_torch_device()
    train_data, val_data = load_splits(config)
    if val_data.features != train_data.features:
        raise ValueError("Train and validation splits declare different features")
    representation = build_representation(config, train_data.features)
    representation.fit(train_data, seed=int(config["seed"]))
    train_data.apply_representation(representation)
    val_data.apply_representation(representation)
    # Rewards are measured in units of the train split's mean |reward|, so the
    # Q-values have a comparable scale on every environment (an Industrial
    # Benchmark step costs about 200; a SimBank step pays 0.7 on average).
    # Only training sees the scaled reward; evaluation reports the raw return.
    reward_scale = train_data.mean_abs_reward()
    train_data.scale_rewards(reward_scale)
    val_data.scale_rewards(reward_scale)
    data_spec = DataSpec(obs_dim=representation.output_dim, num_actions=train_data.num_actions)
    cls = get_algorithm(config["algo"])
    algorithm = cls(config, data_spec, device)

    def snapshot() -> dict[str, Any]:
        # The frozen representation and the network input size travel with
        # every checkpoint, so deployment never refits anything.
        return {
            **algorithm.state_dict(),
            "representation_state": representation.state_dict(),
            "data_spec": {"obs_dim": data_spec.obs_dim, "num_actions": data_spec.num_actions},
        }

    stopping = _EarlyStopping(
        patience=int(config.get("early_stopping_patience", 20)),
        min_delta=float(config.get("early_stopping_min_delta", 0.0)),
    )
    method = str(config.get("method_id", cls.method_id))
    history: list[dict[str, float]] = []
    sampled = 0
    step = 0
    started = time.perf_counter()
    bar = tqdm(range(1, total_steps + 1), desc=method, unit="step", disable=not show_progress)
    batches = algorithm.batches(train_data)
    try:
        for step in bar:
            batch = next(batches)
            sampled += algorithm.batch_size(batch)
            info = algorithm.update(batch, step)
            bar.set_postfix(loss=f"{info['loss']:.4f}", sampled=sampled)
            if step == 1:
                update_progress(
                    status="running", phase="optimization", method_id=method, step=step, total_steps=total_steps
                )
            if step % log_interval != 0 and step != total_steps:
                continue
            val_loss, val_info = algorithm.validate(val_data)
            improved, stop = stopping.check(val_loss, step=step, sampled=sampled, snapshot=snapshot)
            history.append(
                {
                    "step": step,
                    "sampled_transitions": sampled,
                    "train_loss": info["loss"],
                    "validation_loss": val_loss,
                    **{f"train_{key}": value for key, value in info.items() if key != "loss"},
                    **{f"validation_{key}": value for key, value in val_info.items()},
                }
            )
            update_progress(
                status="running",
                phase="optimization",
                method_id=method,
                step=step,
                total_steps=total_steps,
                validation_loss=val_loss,
            )
            if show_progress:
                decision = (
                    "new best"
                    if improved
                    else f"no improvement ({stopping.checks_without_improvement}/{stopping.patience})"
                )
                if stop:
                    decision = "early stop: patience exhausted"
                progress_write(
                    f"[val] step={step:>6d} sampled={sampled:>9d}  train_loss={_format(info['loss'])}  "
                    f"val_loss={_format(val_loss)}  best={_format(stopping.best_loss)}  {decision}"
                )
            if stop:
                break
    finally:
        bar.close()

    effective_steps = int(step)
    summary = {
        "algorithm": cls.name,
        "validation_objective": cls.validation_objective,
        "representation": representation.name,
        "representation_input_dim": representation.input_dim,
        "representation_output_dim": representation.output_dim,
        "reward_scale": reward_scale,
        "offline_train_steps": total_steps,
        "offline_batch_size": int(config["offline_batch_size"]),
        "total_gradient_steps": effective_steps,
        "total_sampled_transitions": int(sampled),
        "effective_epochs": float(sampled / max(1, len(train_data))),
        "train_dataset_transitions": len(train_data),
        "train_dataset_episodes": train_data.num_episodes,
        "val_dataset_transitions": len(val_data),
        "val_dataset_episodes": val_data.num_episodes,
        "best_validation_loss": float(stopping.best_loss),
        "best_validation_step": int(stopping.best_step),
        "best_validation_sampled_transitions": int(stopping.best_sampled),
        "num_validation_checks": int(stopping.checks),
        "early_stopped": stopping.stopped_at is not None,
        "early_stop_step": int(stopping.stopped_at if stopping.stopped_at is not None else effective_steps),
        "early_stop_reason": "patience_exhausted" if stopping.stopped_at is not None else "completed_budget",
        "optimization_seconds": float(time.perf_counter() - started),
    }
    return TrainResult(
        history=history,
        summary=summary,
        best_checkpoint=stopping.best_checkpoint,
        last_checkpoint=snapshot(),
    )
