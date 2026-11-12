"""Contract tests run for every registered algorithm on every environment and representation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import TINY_HYPERPARAMETERS, tiny_run_config

from projeto_rl.algorithms import algorithm_names, get_algorithm, load_policy
from projeto_rl.algorithms.base import td_target
from projeto_rl.data.dataset import OfflineBatch, OfflineTransitionDataset
from projeto_rl.envs import environment_ids
from projeto_rl.evaluation import evaluate_policy
from projeto_rl.operations.artifacts import initial_manifest
from projeto_rl.policies import Policy, masked_uniform
from projeto_rl.representations import representation_names
from projeto_rl.runner import run_experiment
from projeto_rl.training import train_offline

ALGORITHMS = algorithm_names()
CASES = [
    (algo, env_id, representation)
    for algo in ALGORITHMS
    for env_id in environment_ids()
    for representation in representation_names()
]


def test_every_algorithm_has_tiny_hyperparameters():
    assert set(ALGORITHMS) <= set(TINY_HYPERPARAMETERS)


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_registration_metadata(algo):
    cls = get_algorithm(algo)
    assert get_algorithm(cls.method_id) is cls
    assert cls.description and cls.validation_objective


@pytest.mark.parametrize(("algo", "env_id", "representation"), CASES)
def test_train_load_and_act_within_mask(algo, env_id, representation, tiny_datasets):
    config = tiny_run_config(algo, env_id, tiny_datasets[env_id], representation=representation)
    result = train_offline(config)
    assert result.summary["total_gradient_steps"] == config["offline_train_steps"]
    assert result.summary["representation"] == representation
    assert np.isfinite(result.summary["best_validation_loss"])
    assert [row["step"] for row in result.history] == [2, 4]
    policy = load_policy(config, result.best_checkpoint)
    data = OfflineTransitionDataset(tiny_datasets[env_id] / "train.npz")
    policy.reset_episode()
    for index in range(3):
        mask = data.data["action_masks"][index]
        info = {
            "action_mask": mask,
            "admissible_actions": np.flatnonzero(mask).tolist(),
            "state_id": int(data.data["state_ids"][index]),
        }
        obs = data.data["observations"][index]  # raw: the policy applies the representation itself
        probabilities = policy.action_probabilities(obs, info, deterministic=False)
        assert probabilities.shape == mask.shape and np.isclose(probabilities.sum(), 1.0)
        assert np.all(probabilities[mask <= 0] == 0.0)
        assert mask[policy.act(obs, info, deterministic=True)] == 1
        policy.observe(int(data.data["actions"][index]), float(data.data["rewards"][index]))


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_runner_writes_every_artifact(algo, tiny_datasets, tmp_path):
    config = tiny_run_config(
        algo, "epicare", tiny_datasets["epicare"], dataset_id="test/epicare", method_id=get_algorithm(algo).method_id
    )
    manifest = initial_manifest(config, run_dir=tmp_path)
    run_dir = run_experiment(config, output_dir=tmp_path / "run", manifest=manifest, quiet=True)
    for name in (
        "config.json",
        "run_manifest.json",
        "train_history.csv",
        "checkpoint_best.pt",
        "checkpoint_last.pt",
        "eval_summary.json",
        "episodes_summary.csv",
        "trajectories.csv",
        "environment_spec.json",
        "metrics_summary.json",
    ):
        assert (run_dir / name).exists(), name
    assert json.loads((run_dir / "run_manifest.json").read_text())["status"] == "completed"
    summary = json.loads((run_dir / "metrics_summary.json").read_text())
    assert summary["representation"] == "raw" and "mean_return" in summary["eval_summary"]
    assert "validation_loss" in pd.read_csv(run_dir / "train_history.csv").columns
    # A completed run is skipped unless forced.
    assert run_experiment(config, output_dir=run_dir, quiet=True) == run_dir


def test_failed_run_is_recorded(tiny_datasets, tmp_path):
    config = tiny_run_config(
        "dqn",
        "epicare",
        tiny_datasets["epicare"],
        dataset_path=str(tmp_path / "missing.npz"),
        dataset_id="test/epicare",
        method_id="dqn",
    )
    manifest = initial_manifest(config, run_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        run_experiment(config, output_dir=tmp_path / "run", manifest=manifest, quiet=True)
    assert json.loads((tmp_path / "run" / "run_manifest.json").read_text())["status"] == "failed"


def test_early_stopping_selects_first_checkpoint_when_validation_never_improves(tiny_datasets):
    config = tiny_run_config(
        "dqn",
        "epicare",
        tiny_datasets["epicare"],
        learning_rate=0.0,
        offline_train_steps=10,
        offline_log_interval=1,
        early_stopping_patience=1,
    )
    summary = train_offline(config).summary
    assert summary["early_stopped"] and summary["best_validation_step"] == 1 and summary["early_stop_step"] == 2
    assert summary["num_validation_checks"] == 2


def test_simulator_evaluation_is_deterministic(tiny_datasets):
    config = tiny_run_config("dqn", "epicare", tiny_datasets["epicare"], representation="standardize")
    policy = load_policy(config, train_offline(config).best_checkpoint)
    first = evaluate_policy(policy, config=config, num_episodes=3, eval_seed_base=7)
    second = evaluate_policy(policy, config=config, num_episodes=3, eval_seed_base=7)
    pd.testing.assert_frame_equal(first["trajectories"], second["trajectories"])


@pytest.mark.parametrize("env_id", environment_ids())
def test_side_by_side_evaluation_gives_the_episodes_of_sequential_evaluation(tiny_datasets, env_id):
    config = tiny_run_config("dqn", env_id, tiny_datasets[env_id], representation="standardize")
    policy = load_policy(config, train_offline(config).best_checkpoint)
    sequential = evaluate_policy(
        policy, config={**config, "eval_parallel_episodes": 1}, num_episodes=7, eval_seed_base=7
    )
    # Three environments for seven episodes: every environment is reused, and
    # episodes of different lengths start and end at different steps.
    parallel = evaluate_policy(policy, config={**config, "eval_parallel_episodes": 3}, num_episodes=7, eval_seed_base=7)
    pd.testing.assert_frame_equal(parallel["trajectories"], sequential["trajectories"])
    pd.testing.assert_frame_equal(parallel["episodes"], sequential["episodes"])
    assert sequential["episodes"]["episode_id"].tolist() == list(range(7))
    for key, value in sequential["summary"].items():
        assert key == "mean_action_latency_ms" or parallel["summary"][key] == value


class _MaskedUniform(Policy):
    """Uniform over the admissible actions; records how many episodes each ``act_batch`` call served."""

    def __init__(self, *, stateless: bool):
        super().__init__(seed=5)
        self.stateless = stateless
        self.batch_sizes: list[int] = []

    def action_probabilities(self, obs, info, deterministic=True):
        return masked_uniform(info["action_mask"])

    def act_batch(self, observations, infos, deterministic=True):
        self.batch_sizes.append(len(observations))
        return super().act_batch(observations, infos, deterministic=deterministic)


def test_only_stateless_greedy_policies_are_evaluated_side_by_side(tiny_datasets):
    config = {**tiny_run_config("dqn", "epicare", tiny_datasets["epicare"]), "eval_parallel_episodes": 3}

    def batch_sizes(*, stateless: bool, deterministic: bool) -> set[int]:
        policy = _MaskedUniform(stateless=stateless)
        evaluate_policy(policy, config=config, num_episodes=4, eval_seed_base=7, deterministic=deterministic)
        return set(policy.batch_sizes)

    assert max(batch_sizes(stateless=True, deterministic=True)) == 3
    # A policy with per-episode state, or one that draws its actions, would not
    # see the calls in the order of a sequential evaluation.
    assert batch_sizes(stateless=False, deterministic=True) == {1}
    assert batch_sizes(stateless=True, deterministic=False) == {1}
    with pytest.raises(ValueError, match="eval_parallel_episodes must be > 0"):
        evaluate_policy(
            _MaskedUniform(stateless=True),
            config={**config, "eval_parallel_episodes": 0},
            num_episodes=1,
            eval_seed_base=7,
        )


def test_td_target_bootstraps_through_timeouts_but_not_terminals():
    # Three transitions with reward 1 and next value 10, gamma 0.9: an
    # ordinary step and a timeout both bootstrap (1 + 9 = 10); a terminal does not (1).
    batch = OfflineBatch(
        obs=torch.zeros(3, 1),
        action=torch.zeros(3, dtype=torch.int64),
        reward=torch.ones(3),
        next_obs=torch.zeros(3, 1),
        terminal=torch.tensor([0.0, 0.0, 1.0]),
        action_mask=torch.ones(3, 1),
        next_action_mask=torch.ones(3, 1),
    )
    target = td_target(batch, torch.full((3,), 10.0), 0.9)
    torch.testing.assert_close(target, torch.tensor([10.0, 10.0, 1.0]))


def test_reward_scale_is_the_train_mean_absolute_reward(tiny_datasets, tmp_path):
    # Train rewards alternate +2, -2 (mean |r| = 2); validation rewards are 100
    # and must not influence the scale.
    source = tiny_datasets["epicare"]
    for split, pattern in (("train", [2.0, -2.0]), ("val", [100.0])):
        data = dict(np.load(source / f"{split}.npz"))
        data["rewards"] = np.resize(np.asarray(pattern, dtype=np.float32), len(data["rewards"]))
        np.savez(tmp_path / f"{split}.npz", **data)
    (tmp_path / "metadata.json").write_text((source / "metadata.json").read_text())
    train = OfflineTransitionDataset(tmp_path / "train.npz")
    assert train.mean_abs_reward() == 2.0
    train.scale_rewards(2.0)
    np.testing.assert_array_equal(train.data["rewards"][:2], [1.0, -1.0])
    assert train_offline(tiny_run_config("dqn", "epicare", tmp_path)).summary["reward_scale"] == 2.0
    data["rewards"] = np.zeros_like(data["rewards"])
    np.savez(tmp_path / "zero.npz", **data)
    with pytest.raises(ValueError, match="every reward is zero"):
        OfflineTransitionDataset(tmp_path / "zero.npz").mean_abs_reward()
