"""Representations: identity, train-only fitting, frozen state, and deployment through the policy wrapper."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import tiny_dataset_config, tiny_run_config

from projeto_rl.algorithms import load_policy
from projeto_rl.data.collect import collect_dataset
from projeto_rl.data.dataset import OfflineTransitionDataset
from projeto_rl.envs.base import FeatureSpec
from projeto_rl.policies import Policy
from projeto_rl.representations import RepresentationPolicy, build_representation, load_representation
from projeto_rl.training import train_offline

FEATURES = (FeatureSpec("a", "numeric"), FeatureSpec("b", "ordinal", 3))


class _Split:
    """The only attribute representations read from a dataset in these tests."""

    def __init__(self, observations):
        self.observations = np.asarray(observations, dtype=np.float32)


def _fitted(name: str, observations) -> object:
    representation = build_representation({"representation": name}, FEATURES)
    representation.fit(_Split(observations), seed=0)
    return representation


def test_raw_is_the_identity():
    obs = np.array([[1.5, 2.0], [-3.0, 0.0]], dtype=np.float32)
    representation = _fitted("raw", obs)
    np.testing.assert_array_equal(representation.transform(obs), obs)
    assert representation.output_dim == 2


def test_standardize_uses_train_mean_and_population_std():
    train = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 1.0]])
    representation = _fitted("standardize", train)
    # mean (3, 1); population std (sqrt(8/3), sqrt(2/3)).
    expected = np.array([[(7.0 - 3.0) / np.sqrt(8 / 3), (0.0 - 1.0) / np.sqrt(2 / 3)]])
    np.testing.assert_allclose(representation.transform(np.array([[7.0, 0.0]])), expected, rtol=1e-6)
    transformed = representation.transform(train)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-5)


def test_standardize_rejects_constant_columns_and_wrong_shapes():
    with pytest.raises(ValueError, match="constant"):
        _fitted("standardize", [[1.0, 0.0], [1.0, 2.0]])
    representation = _fitted("standardize", [[1.0, 0.0], [3.0, 2.0]])
    with pytest.raises(ValueError, match="shape"):
        representation.transform(np.zeros((2, 3)))


def test_state_dict_round_trip_reproduces_the_transform():
    representation = _fitted("standardize", [[1.0, 0.0], [3.0, 2.0], [4.0, 1.0]])
    restored = load_representation({"representation": "standardize"}, representation.state_dict())
    obs = np.array([[2.0, 1.0], [9.0, 0.0]])
    np.testing.assert_array_equal(restored.transform(obs), representation.transform(obs))
    assert restored.features == FEATURES
    with pytest.raises(ValueError, match="config"):
        load_representation({"representation": "raw"}, representation.state_dict())


def test_training_fits_on_the_train_split_only(tiny_datasets, tmp_path):
    root = tiny_datasets["epicare"]
    config = tiny_run_config("bc", "epicare", root, representation="standardize")
    state = train_offline(config).best_checkpoint["representation_state"]
    train = OfflineTransitionDataset(root / "train.npz").observations.astype(np.float64)
    np.testing.assert_allclose(state["fitted"]["mean"], train.mean(axis=0))
    np.testing.assert_allclose(state["fitted"]["std"], train.std(axis=0))
    # Replacing the validation split must not change the fitted state.
    collect_dataset(tiny_dataset_config("epicare", tmp_path, seed=99))
    swapped = {**config, "val_dataset_path": str(tmp_path / "val.npz")}
    assert train_offline(swapped).best_checkpoint["representation_state"] == state


class _Recorder(Policy):
    def __init__(self):
        super().__init__()
        self.seen = []

    def action_probabilities(self, obs, info, deterministic=True):
        self.seen.append(np.asarray(obs))
        return np.array([1.0, 0.0])


def test_representation_policy_feeds_transformed_observations():
    representation = _fitted("standardize", [[1.0, 0.0], [3.0, 2.0]])
    inner = _Recorder()
    policy = RepresentationPolicy(inner, representation)
    obs = np.array([3.0, 2.0], dtype=np.float32)
    policy.act(obs, {"action_mask": np.ones(2)})
    np.testing.assert_allclose(inner.seen[-1], representation.transform(obs[None, :])[0])
    np.testing.assert_allclose(inner.seen[-1], [1.0, 1.0])


def test_deployed_policy_matches_network_on_transformed_inputs(tiny_datasets):
    config = tiny_run_config("dqn", "epicare", tiny_datasets["epicare"], representation="standardize")
    checkpoint = train_offline(config).best_checkpoint
    policy = load_policy(config, checkpoint)
    assert checkpoint["data_spec"]["obs_dim"] == policy.representation.output_dim
    obs = OfflineTransitionDataset(tiny_datasets["epicare"] / "train.npz").observations[0]
    info = {"action_mask": np.ones(checkpoint["data_spec"]["num_actions"], dtype=np.float32)}
    direct = policy.policy.action_probabilities(policy.representation.transform(obs[None, :])[0], info)
    np.testing.assert_array_equal(policy.action_probabilities(obs, info), direct)
