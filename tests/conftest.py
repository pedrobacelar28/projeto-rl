"""Shared fixtures: tiny collected datasets and tiny run configurations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from projeto_rl.data.collect import collect_dataset
from projeto_rl.paths import CONFIG_ROOT

SIMBANK_KWARGS = json.loads((CONFIG_ROOT / "envs" / "simbank.json").read_text())["run"]["env_kwargs"]

TINY_DATASETS = {
    "epicare": {
        "env_id": "epicare",
        "env_kwargs": {"patient_modifiers": True},
        "environment_seed": 1,
        "gamma": 1.0,
        "behavior_policy": "soc",
        "behavior_epsilon": 0.1,
    },
    "industrial_benchmark": {
        "env_id": "industrial_benchmark",
        "env_kwargs": {"setpoints": [30, 70], "horizon": 12},
        "gamma": 0.99,
        "behavior_policy": "mediocre",
        "behavior_epsilon": 0.1,
        "mediocre_target": 50,
        "mediocre_deadband": 5,
    },
    "sales_promotion": {
        "env_id": "sales_promotion",
        "env_kwargs": {"users": {"collect": [0, 50], "evaluate": [50, 100]}, "horizon": 5},
        "gamma": 1.0,
        "behavior_policy": "ticket_band",
        "behavior_epsilon": 0.1,
        "ticket_band_thresholds": [1.8, 12.0],
        "ticket_band_actions": [[5, 0.7], [3, 0.85], [1, 0.95]],
    },
    "simbank": {
        "env_id": "simbank",
        "env_kwargs": SIMBANK_KWARGS,
        "gamma": 1.0,
        "behavior_policy": "bank",
        "behavior_epsilon": 0.1,
    },
}
ENV_KEYS = {"env_id", "env_kwargs", "environment_seed", "gamma"}

# Small networks and budgets; every algorithm-specific key is listed so the
# tests fail loudly if an algorithm starts reading an undeclared parameter.
TINY_HYPERPARAMETERS = {
    "common": {
        "representation": "raw",
        "offline_train_steps": 4,
        "offline_batch_size": 8,
        "offline_log_interval": 2,
        "learning_rate": 1e-3,
        "max_grad_norm": 10.0,
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 0.0,
        "policy_hidden_dim": 16,
        "policy_hidden_layers": 1,
        "policy_temperature": 1.0,
        "eval_episodes": 2,
        "eval_seed_base": 100_000,
        "eval_deterministic": True,
        "eval_parallel_episodes": 2,
        "save_trajectories": True,
        "show_progress": False,
    },
    "dqn": {"tau": 0.5, "target_network_frequency": 1},
    "bc": {},
    "iql": {"iql_expectile_tau": 0.7, "iql_temperature_beta": 3.0, "iql_advantage_clip": 100.0},
    "cql": {"cql_alpha": 1.0},
}


def tiny_dataset_config(env_id: str, output_dir: Path, *, seed: int = 3) -> dict:
    return {
        **TINY_DATASETS[env_id],
        "seed": seed,
        "num_train_episodes": 6,
        "num_val_episodes": 3,
        "output_dir": str(output_dir),
        "overwrite": True,
        "show_progress": False,
        "quiet": True,
    }


def tiny_run_config(algo: str, env_id: str, dataset_dir: Path, **overrides) -> dict:
    return {
        **{key: value for key, value in TINY_DATASETS[env_id].items() if key in ENV_KEYS},
        **TINY_HYPERPARAMETERS["common"],
        **TINY_HYPERPARAMETERS[algo],
        "algo": algo,
        "seed": 0,
        "dataset_path": str(dataset_dir / "train.npz"),
        "val_dataset_path": str(dataset_dir / "val.npz"),
        **overrides,
    }


@pytest.fixture(scope="session")
def tiny_datasets(tmp_path_factory) -> dict[str, Path]:
    """One tiny collected dataset per environment, shared by the whole session."""

    root = tmp_path_factory.mktemp("datasets")
    paths = {}
    for env_id in TINY_DATASETS:
        paths[env_id] = root / env_id
        collect_dataset(tiny_dataset_config(env_id, paths[env_id]))
    return paths
