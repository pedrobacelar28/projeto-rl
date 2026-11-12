"""Representations: identity, train-only fitting, frozen state, and deployment through the policy wrapper."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import TINY_HYPERPARAMETERS, tiny_dataset_config, tiny_run_config
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabpfn.errors import TabPFNLicenseError

from projeto_rl.algorithms import load_policy
from projeto_rl.data.collect import collect_dataset
from projeto_rl.data.dataset import OfflineTransitionDataset
from projeto_rl.envs.base import FeatureSpec
from projeto_rl.policies import Policy
from projeto_rl.representations import RepresentationPolicy, build_representation, load_representation
from projeto_rl.representations.spr import ema_update, spr_loss
from projeto_rl.representations.tdjepa import (
    TWINS,
    Networks,
    greedy_actions,
    orthonormality_loss,
    sample_tasks,
    td_jepa_loss,
    td_targets,
    to_sphere,
)
from projeto_rl.representations.tjepa import mask_sizes, sample_masks, sincos_positions, tjepa_loss
from projeto_rl.training import train_offline
from projeto_rl.utils.torch_utils import get_torch_device

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


# -- SPR ----------------------------------------------------------------

SPR_FEATURES = tuple(FeatureSpec(name, "numeric") for name in "abc")
SPR_CONFIG = {
    "representation": "spr",
    "spr_latent_dim": 8,
    "spr_hidden_dim": 32,
    "spr_hidden_layers": 1,
    "spr_dropout": 0.5,
    "spr_train_steps": 30,
    "spr_batch_size": 64,
    "spr_learning_rate": 1e-3,
    "spr_target_tau": 0.99,
}


class _Transitions:
    """What SPR reads from the train split."""

    def __init__(self, observations, actions, next_observations, num_actions):
        self.observations = np.asarray(observations, dtype=np.float32)
        self.next_observations = np.asarray(next_observations, dtype=np.float32)
        self.data = {"actions": np.asarray(actions)}
        self.num_actions = num_actions


def _linear_system(seed: int = 0, size: int = 4000) -> _Transitions:
    """A deterministic system: ``s' = M s + (+0.5 if a == 1 else -0.5)``."""

    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(size, 3))
    actions = rng.integers(0, 2, size=size)
    matrix = np.array([[0.9, 0.1, 0.0], [-0.2, 0.8, 0.0], [0.0, 0.0, 0.5]])
    return _Transitions(obs, actions, obs @ matrix.T + np.where(actions[:, None] == 1, 0.5, -0.5), 2)


def _fitted_spr(split, seed: int = 0, **overrides):
    representation = build_representation({**SPR_CONFIG, **overrides}, SPR_FEATURES)
    representation.fit(split, seed=seed)
    return representation


def test_spr_loss_is_the_negative_mean_cosine_similarity():
    # cos([3, 4], [4, 3]) = 24 / 25 and cos([1, 0], [0, 2]) = 0, so the loss is -(0.96 + 0) / 2.
    prediction = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
    target = torch.tensor([[4.0, 3.0], [0.0, 2.0]])
    assert spr_loss(prediction, target).item() == pytest.approx(-0.48)
    assert spr_loss(prediction, 7.0 * prediction).item() == pytest.approx(-1.0)


def test_spr_target_update_is_an_exponential_moving_average():
    target, online = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    with torch.no_grad():
        target.weight.fill_(1.0), target.bias.fill_(0.0)
        online.weight.fill_(3.0), online.bias.fill_(-2.0)
    ema_update(target, online, 0.9)
    # 0.9 * 1 + 0.1 * 3 = 1.2 and 0.9 * 0 + 0.1 * (-2) = -0.2; the online network is untouched.
    assert target.weight.item() == pytest.approx(1.2) and target.bias.item() == pytest.approx(-0.2)
    assert online.weight.item() == 3.0 and online.bias.item() == -2.0


def test_spr_transform_is_the_frozen_encoder_on_raw_inputs():
    split = _linear_system()
    representation = _fitted_spr(split)
    state = representation.state_dict()["fitted"]
    assert sorted(state) == ["encoder", "latent_mean", "latent_std"]
    # The forward pass by hand, from the stored weights: one ReLU layer on the
    # raw observation, a linear layer, then the latent z-score.
    weights = {key: value.numpy().astype(np.float64) for key, value in state["encoder"].items()}
    obs = np.array([[0.3, -1.2, 2.0], [5.0, 0.0, -0.7]])
    hidden = np.maximum(obs @ weights["layers.0.weight"].T + weights["layers.0.bias"], 0.0)
    latent = hidden @ weights["layers.1.weight"].T + weights["layers.1.bias"]
    expected = (latent - state["latent_mean"]) / state["latent_std"]
    np.testing.assert_allclose(representation.transform(obs), expected, rtol=1e-4, atol=1e-5)
    assert representation.output_dim == 8
    transformed = representation.transform(split.observations)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-4)


def test_spr_reaches_the_loss_floor_on_a_deterministic_system_and_keeps_the_state():
    # The next latent is a function of (state, action), so without dropout
    # noise the prediction can be exact: the cosine loss goes to its floor, -1.
    split = _linear_system()
    representation = _fitted_spr(split, spr_dropout=0.0, spr_train_steps=300)
    history = representation.loss_history
    assert len(history) == 300 and np.mean(history[-50:]) < -0.99 < np.mean(history[:50])
    # The latent has not collapsed: every observation column is linearly decodable from it.
    latent = np.c_[representation.transform(split.observations), np.ones(len(split.observations))]
    inputs = split.observations.astype(np.float64)
    residual = inputs - latent @ np.linalg.lstsq(latent, inputs, rcond=None)[0]
    assert np.all(1.0 - residual.var(axis=0) / inputs.var(axis=0) > 0.9)


def test_spr_fit_is_seeded_and_leaves_the_global_generators_alone():
    split = _linear_system()
    torch_state, numpy_state = torch.get_rng_state().clone(), np.random.get_state()[1].copy()
    first, second, other = _fitted_spr(split, seed=0), _fitted_spr(split, seed=0), _fitted_spr(split, seed=1)
    assert torch.equal(torch.get_rng_state(), torch_state)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state)
    obs = split.observations[:5]
    np.testing.assert_array_equal(first.transform(obs), second.transform(obs))
    assert first.loss_history == second.loss_history
    assert not np.allclose(first.transform(obs), other.transform(obs))


def test_spr_state_round_trip_through_a_checkpoint_file(tmp_path):
    split = _linear_system()
    representation = _fitted_spr(split)
    torch.save({"representation_state": representation.state_dict()}, tmp_path / "checkpoint.pt")
    state = torch.load(tmp_path / "checkpoint.pt", map_location="cpu")["representation_state"]
    restored = load_representation(SPR_CONFIG, state)
    np.testing.assert_array_equal(restored.transform(split.observations), representation.transform(split.observations))
    # A checkpoint never loads into an encoder of another size.
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_representation({**SPR_CONFIG, "spr_latent_dim": 4}, state)


def test_spr_rejects_missing_settings_and_non_finite_latents():
    with pytest.raises(ValueError, match="spr_latent_dim"):
        build_representation({"representation": "spr"}, SPR_FEATURES)
    with pytest.raises(ValueError, match="spr_target_tau"):
        _fitted_spr(_linear_system(), spr_target_tau=1.0)
    split = _linear_system()
    split.observations[0, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        _fitted_spr(split)


def test_training_fits_spr_on_the_train_split_only(tiny_datasets, tmp_path):
    root = tiny_datasets["epicare"]
    config = tiny_run_config("bc", "epicare", root, representation="spr")
    state = train_offline(config).best_checkpoint["representation_state"]["fitted"]
    assert state["encoder"]["layers.1.weight"].shape == (TINY_HYPERPARAMETERS["common"]["spr_latent_dim"], 8)
    # Replacing the validation split must not change the fitted encoder.
    collect_dataset(tiny_dataset_config("epicare", tmp_path, seed=99))
    swapped = {**config, "val_dataset_path": str(tmp_path / "val.npz")}
    other = train_offline(swapped).best_checkpoint["representation_state"]["fitted"]
    assert all(torch.equal(other["encoder"][key], value) for key, value in state["encoder"].items())
    assert other["latent_mean"] == state["latent_mean"] and other["latent_std"] == state["latent_std"]


# -- T-JEPA -------------------------------------------------------------

TJEPA_FEATURES = (
    FeatureSpec("a", "numeric"),
    FeatureSpec("b", "numeric"),
    FeatureSpec("c", "categorical", 3),
    FeatureSpec("d", "numeric"),
)
TJEPA_CONFIG = {
    "representation": "tjepa",
    "tjepa_hidden_dim": 8,
    "tjepa_num_layers": 1,
    "tjepa_num_heads": 2,
    "tjepa_feedforward_dim": 16,
    "tjepa_predictor_dim": 8,
    "tjepa_predictor_layers": 1,
    "tjepa_predictor_heads": 2,
    "tjepa_min_context_share": 0.2,
    "tjepa_max_context_share": 0.4,
    "tjepa_min_target_share": 0.2,
    "tjepa_max_target_share": 0.4,
    "tjepa_target_masks": 2,
    "tjepa_train_steps": 30,
    "tjepa_batch_size": 64,
    "tjepa_learning_rate": 1e-3,
    "tjepa_ema_start": 0.996,
    "tjepa_ema_end": 1.0,
}


def _table(seed: int = 0, size: int = 4000) -> _Split:
    """Rows that are functions of one latent ``a``: ``10 + 3a``, ``5 - 2a``, the band of ``a``, ``100 tanh(a)``."""

    a = np.random.default_rng(seed).normal(size=size)
    return _Split(np.c_[10.0 + 3.0 * a, 5.0 - 2.0 * a, np.digitize(a, [-0.5, 0.5]), 100.0 * np.tanh(a)])


def _fitted_tjepa(split, seed: int = 0, **overrides):
    representation = build_representation({**TJEPA_CONFIG, **overrides}, TJEPA_FEATURES)
    representation.fit(split, seed=seed)
    return representation


def _layer_norm(values, weight, bias):
    centered = values - values.mean(axis=-1, keepdims=True)
    return centered / np.sqrt((centered**2).mean(axis=-1, keepdims=True) + 1e-5) * weight + bias


def _encoder_layer(tokens, weights, prefix: str, heads: int):
    """One post-norm transformer encoder layer in numpy: ``[B, N, h]`` to ``[B, N, h]``."""

    dim = tokens.shape[-1]
    projection, offset = weights[f"{prefix}.self_attn.in_proj_weight"], weights[f"{prefix}.self_attn.in_proj_bias"]
    query, key, value = (
        (tokens @ projection[part * dim : (part + 1) * dim].T + offset[part * dim : (part + 1) * dim])
        .reshape(*tokens.shape[:2], heads, dim // heads)
        .transpose(0, 2, 1, 3)
        for part in range(3)
    )
    scores = query @ key.transpose(0, 1, 3, 2) / np.sqrt(dim // heads)
    attention = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attention /= attention.sum(axis=-1, keepdims=True)
    mixed = (attention @ value).transpose(0, 2, 1, 3).reshape(tokens.shape)
    mixed = mixed @ weights[f"{prefix}.self_attn.out_proj.weight"].T + weights[f"{prefix}.self_attn.out_proj.bias"]
    tokens = _layer_norm(tokens + mixed, weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"])
    hidden = np.maximum(tokens @ weights[f"{prefix}.linear1.weight"].T + weights[f"{prefix}.linear1.bias"], 0.0)
    hidden = hidden @ weights[f"{prefix}.linear2.weight"].T + weights[f"{prefix}.linear2.bias"]
    return _layer_norm(tokens + hidden, weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"])


def test_tjepa_loss_is_the_mean_squared_error():
    # Squared errors 1, 0, 0, 4 over four entries.
    prediction = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    target = torch.tensor([[[0.0, 2.0], [3.0, 6.0]]])
    assert tjepa_loss(prediction, target).item() == pytest.approx(1.25)


def test_tjepa_positions_are_the_sine_cosine_embedding():
    # Frequencies 10000^0 = 1 and 10000^(-1/2) = 0.01: [sin(p), sin(0.01 p), cos(p), cos(0.01 p)].
    expected = [[np.sin(p), np.sin(0.01 * p), np.cos(p), np.cos(0.01 * p)] for p in range(3)]
    np.testing.assert_allclose(sincos_positions(3, 4).numpy(), expected, atol=1e-7)


def test_tjepa_masks_hold_a_share_of_the_columns_and_never_overlap_the_context():
    # 20% to 40% of the columns, rounded: 1 to 2 of 5, 1 to 2 of 6, 2 to 3 of 8.
    assert [mask_sizes(columns, 0.2, 0.4) for columns in (5, 6, 8)] == [(1, 2), (1, 2), (2, 3)]
    context, targets = sample_masks(np.random.default_rng(0), 200, 8, 3, 2, 4)
    assert context.shape == (200, 3) and targets.shape == (4, 200, 2)
    for row in range(200):
        assert len(set(context[row])) == 3 and set(context[row]) <= set(range(8))
        for target in targets[:, row]:
            assert len(set(target)) == 2 and set(target) <= set(range(8)) - set(context[row])
    # Every column is used on both sides, and the draw is seeded.
    assert set(context.ravel()) == set(targets.ravel()) == set(range(8))
    again = sample_masks(np.random.default_rng(0), 200, 8, 3, 2, 4)
    np.testing.assert_array_equal(context, again[0]), np.testing.assert_array_equal(targets, again[1])


def test_tjepa_transform_is_the_frozen_context_encoder_on_standardized_columns():
    split = _table()
    representation = _fitted_tjepa(split)
    state = representation.state_dict()["fitted"]
    assert sorted(state) == ["encoder", "mean", "std"]
    # Numeric columns carry the train mean and population std; the coded column is left alone.
    train = split.observations.astype(np.float64)
    np.testing.assert_allclose(state["mean"], [train[:, 0].mean(), train[:, 1].mean(), 0.0, train[:, 3].mean()])
    np.testing.assert_allclose(state["std"], [train[:, 0].std(), train[:, 1].std(), 1.0, train[:, 3].std()])
    # The forward pass by hand, from the stored weights: one token per column
    # (scalar * weight + bias, or the embedding row of the code), the [REG]
    # token appended, one transformer layer, [REG] dropped, tokens flattened.
    weights = {key: value.numpy().astype(np.float64) for key, value in state["encoder"].items()}
    obs = np.array([[12.0, 1.0, 2.0, 30.0], [4.0, 9.0, 0.0, 250.0]])
    scaled = (obs - state["mean"]) / state["std"]
    tokens = np.stack(
        [
            scaled[:, [0]] * weights["tokenizer.columns.0.weight"].T + weights["tokenizer.columns.0.bias"],
            scaled[:, [1]] * weights["tokenizer.columns.1.weight"].T + weights["tokenizer.columns.1.bias"],
            weights["tokenizer.columns.2.weight"][[2, 0]],
            scaled[:, [3]] * weights["tokenizer.columns.3.weight"].T + weights["tokenizer.columns.3.bias"],
            np.tile(weights["tokenizer.reg_token"], (2, 1)),
        ],
        axis=1,
    )
    expected = _encoder_layer(tokens, weights, "transformer.layers.0", heads=2)[:, :4].reshape(2, 32)
    np.testing.assert_allclose(representation.transform(obs), expected, rtol=1e-4, atol=1e-5)
    assert representation.output_dim == 32


def test_tjepa_learns_to_predict_masked_columns_without_collapsing():
    split = _table()
    representation = _fitted_tjepa(split, tjepa_train_steps=400, tjepa_learning_rate=3e-3)
    history = representation.loss_history
    # At the start the predictor outputs about zero and the targets are
    # layer-normalized tokens (zero mean, unit variance), so the loss is 1.
    # Every column is a function of the others, so the predictor ends far below it.
    assert len(history) == 400 and history[0] == pytest.approx(1.0, abs=0.05)
    assert np.mean(history[-50:]) < 0.3
    # The representation has not collapsed: every numeric column is linearly decodable from it.
    latent = np.c_[representation.transform(split.observations), np.ones(len(split.observations))]
    inputs = split.observations[:, [0, 1, 3]].astype(np.float64)
    residual = inputs - latent @ np.linalg.lstsq(latent, inputs, rcond=None)[0]
    assert np.all(1.0 - residual.var(axis=0) / inputs.var(axis=0) > 0.9)


def test_tjepa_fit_is_seeded_and_leaves_the_global_generators_alone():
    split = _table()
    torch_state, numpy_state = torch.get_rng_state().clone(), np.random.get_state()[1].copy()
    first, second, other = _fitted_tjepa(split, seed=0), _fitted_tjepa(split, seed=0), _fitted_tjepa(split, seed=1)
    restored = load_representation(TJEPA_CONFIG, first.state_dict())
    assert torch.equal(torch.get_rng_state(), torch_state)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state)
    obs = split.observations[:5]
    np.testing.assert_array_equal(first.transform(obs), second.transform(obs))
    assert first.loss_history == second.loss_history
    assert not np.allclose(first.transform(obs), other.transform(obs))
    np.testing.assert_allclose(restored.transform(obs), first.transform(obs), atol=1e-5)


def test_tjepa_state_round_trip_through_a_checkpoint_file(tmp_path):
    split = _table()
    representation = _fitted_tjepa(split)
    torch.save({"representation_state": representation.state_dict()}, tmp_path / "checkpoint.pt")
    state = torch.load(tmp_path / "checkpoint.pt", map_location="cpu")["representation_state"]
    restored = load_representation(TJEPA_CONFIG, state)
    # The restored encoder runs on the CPU; the fitted one may be on the GPU.
    np.testing.assert_allclose(
        restored.transform(split.observations), representation.transform(split.observations), atol=1e-5
    )
    # A checkpoint never loads into an encoder of another size.
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_representation({**TJEPA_CONFIG, "tjepa_hidden_dim": 4}, state)


def test_tjepa_rejects_bad_settings_and_bad_columns():
    with pytest.raises(ValueError, match="tjepa_hidden_dim"):
        build_representation({"representation": "tjepa"}, TJEPA_FEATURES)
    # 5% of four columns rounds to no column at all.
    with pytest.raises(ValueError, match="at least one column"):
        build_representation({**TJEPA_CONFIG, "tjepa_min_target_share": 0.05}, TJEPA_FEATURES)
    # Three context columns and three target columns do not fit in four.
    with pytest.raises(ValueError, match="do not fit"):
        build_representation(
            {**TJEPA_CONFIG, "tjepa_max_context_share": 0.8, "tjepa_max_target_share": 0.8}, TJEPA_FEATURES
        )
    with pytest.raises(ValueError, match="tjepa_predictor_dim must be even"):
        build_representation({**TJEPA_CONFIG, "tjepa_predictor_dim": 7}, TJEPA_FEATURES)
    with pytest.raises(ValueError, match="tjepa_ema_start"):
        _fitted_tjepa(_table(), tjepa_ema_start=1.0, tjepa_ema_end=0.9)
    constant = _table()
    constant.observations[:, 1] = 3.0
    with pytest.raises(ValueError, match=r"constant train columns: \['b'\]"):
        _fitted_tjepa(constant)
    broken = _table()
    broken.observations[0, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _fitted_tjepa(broken)


def test_training_fits_tjepa_on_the_train_split_only(tiny_datasets, tmp_path):
    # SimBank has ordinal and categorical columns, so the coded tokens go through the whole pipeline.
    root = tiny_datasets["simbank"]
    config = tiny_run_config("bc", "simbank", root, representation="tjepa")
    result = train_offline(config)
    state = result.best_checkpoint["representation_state"]["fitted"]
    hidden = TINY_HYPERPARAMETERS["common"]["tjepa_hidden_dim"]
    assert result.summary["representation_output_dim"] == result.summary["representation_input_dim"] * hidden
    assert state["encoder"]["tokenizer.reg_token"].shape == (hidden,)
    # Replacing the validation split must not change the fitted encoder.
    collect_dataset(tiny_dataset_config("simbank", tmp_path, seed=99))
    swapped = {**config, "val_dataset_path": str(tmp_path / "val.npz")}
    other = train_offline(swapped).best_checkpoint["representation_state"]["fitted"]
    assert all(torch.equal(other["encoder"][key], value) for key, value in state["encoder"].items())
    assert other["mean"] == state["mean"] and other["std"] == state["std"]


# -- TD-JEPA ------------------------------------------------------------------

TDJEPA_FEATURES = tuple(FeatureSpec(name, "numeric") for name in "abc")
TDJEPA_CONFIG = {
    "representation": "tdjepa",
    "tdjepa_state_dim": 8,
    "tdjepa_task_dim": 4,
    "tdjepa_encoder_hidden_dim": 16,
    "tdjepa_encoder_hidden_layers": 2,
    "tdjepa_predictor_hidden_dim": 32,
    "tdjepa_discount": 0.9,
    "tdjepa_orthonormality_coef": 1.0,
    "tdjepa_goal_share": 0.5,
    "tdjepa_train_steps": 30,
    "tdjepa_batch_size": 64,
    "tdjepa_learning_rate": 1e-3,
    "tdjepa_ema": 0.99,
}


def _logged_system() -> _Transitions:
    """The deterministic linear system, with the terminal flags and action masks TD-JEPA also reads.

    An episode ends when the third column of the next state exceeds 0.5 (about
    a quarter of the rows), so the flag is a function of (state, action) too.
    """

    split = _linear_system()
    split.data["terminals"] = (split.next_observations[:, 2] > 0.5).astype(np.float32)
    split.data["next_action_masks"] = np.ones((len(split.observations), 2), dtype=np.float32)
    return split


def _fitted_tdjepa(split, seed: int = 0, **overrides):
    representation = build_representation({**TDJEPA_CONFIG, **overrides}, TDJEPA_FEATURES)
    representation.fit(split, seed=seed)
    return representation


def _tdjepa_networks(seed: int = 0) -> tuple[Networks, Networks]:
    """Online and target networks with different weights: 3 columns, 3 actions, 8 state and 4 task dimensions."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Networks(3, 3, 8, 4, 16, 2, 32), Networks(3, 3, 8, 4, 16, 2, 32)


def test_tdjepa_losses_match_hand_computed_values():
    # Gram matrix [[1, 0, 1], [0, 1, 1], [1, 1, 2]]: the six off-diagonal
    # squares sum to 4, so (1/2) * 4 / 6 = 1/3; the diagonal averages 4/3.
    latent = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert orthonormality_loss(latent).item() == pytest.approx(1.0 / 3.0 - 4.0 / 3.0)
    # Squared distances: twin 0 gives 1 and 1, twin 1 gives 5 and 4; their mean is 2.75.
    prediction = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[1.0, 2.0], [3.0, 0.0]]])
    target = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    assert td_jepa_loss(prediction, target).item() == pytest.approx(2.75)
    np.testing.assert_allclose(to_sphere(torch.tensor([[3.0, 0.0, 4.0, 0.0]])).numpy(), [[1.2, 0.0, 1.6, 0.0]])


def test_tdjepa_tasks_are_on_the_sphere_and_mix_goals_with_uniform_draws():
    goals = to_sphere(torch.randn((4000, 4), generator=torch.Generator().manual_seed(0)))

    def tasks(share: float) -> torch.Tensor:
        return sample_tasks(goals, share, torch.Generator().manual_seed(1))

    def is_goal(z: torch.Tensor) -> torch.Tensor:
        return torch.isclose(z[:, None, :], goals[None, :, :], rtol=1e-5).all(dim=-1).any(dim=1)

    for share in (0.0, 0.5, 1.0):
        np.testing.assert_allclose(tasks(share).norm(dim=1).numpy(), 2.0, rtol=1e-5)
    # Every row is another row's goal, each goal used once.
    np.testing.assert_allclose(tasks(1.0).sort(dim=0).values.numpy(), goals.sort(dim=0).values.numpy(), rtol=1e-5)
    assert not is_goal(tasks(0.0)).any()
    assert is_goal(tasks(0.5)).float().mean().item() == pytest.approx(0.5, abs=0.03)
    # Uniform on the sphere: no preferred direction, unit mean square per entry.
    uniform = tasks(0.0)
    np.testing.assert_allclose(uniform.mean(dim=0).numpy(), 0.0, atol=0.06)
    np.testing.assert_allclose(uniform.pow(2).mean(dim=0).numpy(), 1.0, atol=0.06)
    # A goal that is not on the sphere is put on it.
    scaled = sample_tasks(3.0 * goals, 1.0, torch.Generator().manual_seed(1))
    np.testing.assert_allclose(scaled.numpy(), tasks(1.0).numpy(), rtol=1e-5)


def test_tdjepa_greedy_action_maximizes_the_twin_mean_value_over_the_valid_actions():
    # One row, three actions, two dimensions. Twin means: [2, 0], [0, 3], [1, 1].
    predictions = torch.tensor([[[[4.0, 0.0], [0.0, 2.0], [1.0, 1.0]]], [[[0.0, 0.0], [0.0, 4.0], [1.0, 1.0]]]])
    everything, without_last = torch.ones((1, 3)), torch.tensor([[1.0, 1.0, 0.0]])
    # z = [1, 1] values the actions 2, 3, 2; z = [1, -1] values them 2, -3, 0.
    assert greedy_actions(predictions, torch.tensor([[1.0, 1.0]]), everything).tolist() == [1]
    assert greedy_actions(predictions, torch.tensor([[1.0, -1.0]]), everything).tolist() == [0]
    # z = [-1, 0.5] values them -2, 1.5, -0.5 and z = [-1, 0] values them -2, 0, -1.
    assert greedy_actions(predictions, torch.tensor([[-1.0, 0.5]]), without_last).tolist() == [1]
    assert greedy_actions(predictions, torch.tensor([[-1.0, 0.0]]), torch.tensor([[1.0, 0.0, 1.0]])).tolist() == [2]


def test_tdjepa_targets_follow_the_bellman_equation_of_the_paper():
    online, target = _tdjepa_networks()
    generator = torch.Generator().manual_seed(0)
    next_obs = torch.randn((6, 3), generator=generator)
    z = to_sphere(torch.randn((6, 4), generator=generator))
    continuing = torch.tensor([0.9, 0.9, 0.0, 0.9, 0.0, 0.9])[:, None]
    with torch.no_grad():
        values = (online.state_predictor.all_actions(target.state_encoder(next_obs), z).mean(dim=0) * z[:, None]).sum(
            dim=-1
        )
    # Rows 0 and 1 may not take the action their policy prefers.
    mask = torch.ones((6, 3))
    mask[[0, 1], values[:2].argmax(dim=1)] = 0.0
    state_target, task_target = td_targets(online, target, next_obs, z, mask, continuing)

    actions = torch.eye(3)
    for row in range(6):
        obs, task = next_obs[row : row + 1], z[row : row + 1]
        with torch.no_grad():
            next_state, next_task = target.state_encoder(obs), target.task_encoder(obs)
            # The policy of task z: the valid action with the largest T_phi . z, read from the online predictor.
            value = [
                (online.state_predictor(next_state, task, actions[a : a + 1]).mean(dim=0) @ task.T).item()
                for a in range(3)
            ]
            chosen = max((a for a in range(3) if mask[row, a] > 0), key=lambda a: value[a])
            successor_task = target.state_predictor(next_state, task, actions[chosen : chosen + 1]).mean(dim=0)
            successor_state = target.task_predictor(next_task, task, actions[chosen : chosen + 1]).mean(dim=0)
        gamma = continuing[row].item()
        np.testing.assert_allclose(state_target[row].numpy(), (next_task + gamma * successor_task)[0], atol=1e-5)
        np.testing.assert_allclose(task_target[row].numpy(), (next_state + gamma * successor_state)[0], atol=1e-5)
        if row < 2:
            assert chosen != int(np.argmax(value))
    # A terminal row has no future: its targets are the target encoders' latents of the next state.
    with torch.no_grad():
        np.testing.assert_allclose(state_target[2].numpy(), target.task_encoder(next_obs[2:3])[0].numpy(), atol=1e-6)
        np.testing.assert_allclose(task_target[4].numpy(), target.state_encoder(next_obs[4:5])[0].numpy(), atol=1e-6)
    assert state_target.shape == (6, 4) and task_target.shape == (6, 8)


def test_tdjepa_predictors_are_twins_that_read_the_action_one_hot():
    online, _ = _tdjepa_networks()
    generator = torch.Generator().manual_seed(0)
    latent, z = torch.randn((5, 8), generator=generator), torch.randn((5, 4), generator=generator)
    with torch.no_grad():
        every = online.state_predictor.all_actions(latent, z)
        assert every.shape == (TWINS, 5, 3, 4)
        for action in range(3):
            one_hot = torch.eye(3)[action].expand(5, 3)
            np.testing.assert_allclose(every[:, :, action], online.state_predictor(latent, z, one_hot), atol=1e-6)
        assert online.task_predictor(z, z, torch.eye(3)[[0, 1, 2, 0, 1]]).shape == (TWINS, 5, 8)
    assert not torch.allclose(every[0], every[1])


def test_tdjepa_transform_is_the_frozen_state_encoder_on_standardized_columns():
    split = _logged_system()
    representation = _fitted_tdjepa(split)
    state = representation.state_dict()["fitted"]
    assert sorted(state) == ["encoder", "scaler"]
    weights = {key: value.numpy().astype(np.float64) for key, value in state["encoder"].items()}
    obs = split.observations[:7].astype(np.float64)
    np.testing.assert_allclose(state["scaler"]["mean"], split.observations.astype(np.float64).mean(axis=0))
    scaled = (obs - np.asarray(state["scaler"]["mean"])) / np.asarray(state["scaler"]["std"])
    # Linear, layer normalization, tanh; linear, ReLU; linear; then the sphere of radius sqrt(8).
    hidden = scaled @ weights["network.0.weight"].T + weights["network.0.bias"]
    hidden = (hidden - hidden.mean(axis=1, keepdims=True)) / np.sqrt(hidden.var(axis=1, keepdims=True) + 1e-5)
    hidden = np.tanh(hidden * weights["network.1.weight"] + weights["network.1.bias"])
    hidden = np.maximum(hidden @ weights["network.3.weight"].T + weights["network.3.bias"], 0.0)
    output = hidden @ weights["network.5.weight"].T + weights["network.5.bias"]
    expected = np.sqrt(8.0) * output / np.linalg.norm(output, axis=1, keepdims=True)
    transformed = representation.transform(split.observations[:7])
    np.testing.assert_allclose(transformed, expected, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(transformed, axis=1), np.sqrt(8.0), rtol=1e-5)
    assert representation.output_dim == 8


def _second_moment_spectrum(representation, split) -> np.ndarray:
    latent = representation.transform(split.observations).astype(np.float64)
    return np.linalg.eigvalsh(latent.T @ latent / len(latent))


@pytest.mark.slow
def test_tdjepa_learns_to_predict_the_next_latents_of_a_deterministic_system_without_collapsing():
    split = _logged_system()
    settings = {"tdjepa_train_steps": 1200, "tdjepa_batch_size": 256, "tdjepa_discount": 0.5}
    representation = _fitted_tdjepa(split, **settings)
    history = representation.loss_history
    # The next state, and whether it is terminal, are functions of (state,
    # action), so both targets can be predicted exactly. The loss starts near
    # the spread of latents on the two spheres (8 + 4 per row) and ends far
    # below it; a short horizon lets the bootstrapped part settle in this
    # many steps.
    assert len(history) == 1200 and np.mean(history[:20]) > 6.0
    assert np.mean(history[-50:]) < 2.0
    # The regularizer is minimal when E[phi phi^T] is the identity: every
    # direction of the latent carries the same share of the squared norm.
    spectrum = _second_moment_spectrum(representation, split)
    assert spectrum.min() > 0.8 and spectrum.max() < 1.25
    # Without it the prediction is easier and the latent lies mostly along one direction.
    unregularized = _fitted_tdjepa(split, tdjepa_orthonormality_coef=0.0, **settings)
    assert np.mean(unregularized.loss_history[-50:]) < np.mean(history[-50:])
    assert _second_moment_spectrum(unregularized, split).max() > 4.0


def test_tdjepa_fit_is_seeded_and_leaves_the_global_generators_alone():
    split = _logged_system()
    torch_state, numpy_state = torch.get_rng_state().clone(), np.random.get_state()[1].copy()
    first, second, other = _fitted_tdjepa(split, seed=0), _fitted_tdjepa(split, seed=0), _fitted_tdjepa(split, seed=1)
    restored = load_representation(TDJEPA_CONFIG, first.state_dict())
    assert torch.equal(torch.get_rng_state(), torch_state)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state)
    obs = split.observations[:5]
    np.testing.assert_array_equal(first.transform(obs), second.transform(obs))
    assert first.loss_history == second.loss_history
    assert not np.allclose(first.transform(obs), other.transform(obs))
    np.testing.assert_array_equal(restored.transform(obs), first.transform(obs))


def test_tdjepa_state_round_trip_through_a_checkpoint_file(tmp_path):
    split = _logged_system()
    representation = _fitted_tdjepa(split)
    torch.save({"representation_state": representation.state_dict()}, tmp_path / "checkpoint.pt")
    state = torch.load(tmp_path / "checkpoint.pt", map_location="cpu")["representation_state"]
    restored = load_representation(TDJEPA_CONFIG, state)
    np.testing.assert_array_equal(restored.transform(split.observations), representation.transform(split.observations))
    # A checkpoint never loads into an encoder of another size.
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_representation({**TDJEPA_CONFIG, "tdjepa_state_dim": 4}, state)


def test_tdjepa_rejects_bad_settings_and_bad_columns():
    with pytest.raises(ValueError, match="tdjepa_state_dim"):
        build_representation({"representation": "tdjepa"}, TDJEPA_FEATURES)
    with pytest.raises(ValueError, match="must be > 0"):
        build_representation({**TDJEPA_CONFIG, "tdjepa_encoder_hidden_layers": 0}, TDJEPA_FEATURES)
    with pytest.raises(ValueError, match="tdjepa_predictor_hidden_dim must be even"):
        build_representation({**TDJEPA_CONFIG, "tdjepa_predictor_hidden_dim": 7}, TDJEPA_FEATURES)
    split = _logged_system()
    # An undiscounted sum of latents has no bound; one row leaves no pair for the regularizer.
    with pytest.raises(ValueError, match="tdjepa_discount"):
        _fitted_tdjepa(split, tdjepa_discount=1.0)
    with pytest.raises(ValueError, match="tdjepa_batch_size >= 2"):
        _fitted_tdjepa(split, tdjepa_batch_size=1)
    with pytest.raises(ValueError, match="tdjepa_goal_share"):
        _fitted_tdjepa(split, tdjepa_goal_share=1.5)
    constant = _logged_system()
    constant.observations[:, 1] = 3.0
    with pytest.raises(ValueError, match=r"constant train columns: \['b'\]"):
        _fitted_tdjepa(constant)
    broken = _logged_system()
    broken.next_observations[0, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite values in the next observations"):
        _fitted_tdjepa(broken)


def test_training_fits_tdjepa_on_the_train_split_only(tiny_datasets, tmp_path):
    root = tiny_datasets["simbank"]
    config = tiny_run_config("bc", "simbank", root, representation="tdjepa")
    result = train_offline(config)
    state = result.best_checkpoint["representation_state"]["fitted"]
    common = TINY_HYPERPARAMETERS["common"]
    assert result.summary["representation_output_dim"] == common["tdjepa_state_dim"]
    assert state["encoder"]["network.0.weight"].shape == (
        common["tdjepa_encoder_hidden_dim"],
        result.summary["representation_input_dim"],
    )
    # Replacing the validation split must not change the fitted encoder.
    collect_dataset(tiny_dataset_config("simbank", tmp_path, seed=99))
    swapped = {**config, "val_dataset_path": str(tmp_path / "val.npz")}
    other = train_offline(swapped).best_checkpoint["representation_state"]["fitted"]
    assert all(torch.equal(other["encoder"][key], value) for key, value in state["encoder"].items())
    assert other["scaler"] == state["scaler"]


# -- TabPFN -------------------------------------------------------------------

TABPFN_FEATURES = (FeatureSpec("a", "numeric"), FeatureSpec("b", "numeric"), FeatureSpec("c", "categorical", 3))
# The 2.0 weights download without a license; the experiment uses the version in configs/protocol.json.
TABPFN_CONFIG = {
    "representation": "tabpfn",
    "tabpfn_version": "v2",
    "tabpfn_fallback_version": "v2",
    "tabpfn_context_size": 40,
}


class _LoggedSplit(_Split):
    """A split with the logged actions, which TabPFN takes as the labels of its context."""

    def __init__(self, observations, actions):
        super().__init__(observations)
        self.data = {"actions": np.asarray(actions, dtype=np.int64)}


def _logged_table(rows: int = 120) -> _LoggedSplit:
    rng = np.random.default_rng(11)
    a, b, c = rng.normal(size=rows), rng.normal(size=rows), rng.integers(0, 3, size=rows)
    # Three actions that depend on every column.
    return _LoggedSplit(np.stack([10.0 + 3.0 * a, b, c], axis=1), (a + b > 0.0).astype(np.int64) + (c == 2))


def _fitted_tabpfn(split, *, seed: int = 0, **overrides):
    representation = build_representation({**TABPFN_CONFIG, **overrides}, TABPFN_FEATURES)
    representation.fit(split, seed=seed)
    return representation


@pytest.mark.slow
def test_tabpfn_transform_is_the_z_scored_embedding_of_the_source_model_given_the_sampled_context():
    split = _logged_table()
    representation = _fitted_tabpfn(split, seed=3)
    # The source package driven directly, with the context this seed draws from the train split.
    rows = np.sort(np.random.default_rng(3).choice(120, size=40, replace=False))
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        n_estimators=1,
        fit_mode="fit_with_cache",
        device=str(get_torch_device()),
        random_state=3,
        inference_precision=torch.float32,
        categorical_features_indices=[2],
    )
    model.fit(split.observations[rows], split.data["actions"][rows])
    embedding = np.asarray(model.get_embeddings(split.observations, data_source="test")).reshape(120, -1)
    embedding = embedding.astype(np.float64)
    out = representation.transform(split.observations)
    assert representation.output_dim == embedding.shape[1] == 192
    np.testing.assert_allclose(out, (embedding - embedding.mean(axis=0)) / embedding.std(axis=0), atol=1e-4)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-4)
    state = representation.fitted_state()
    assert sorted(state) == ["context", "labels", "mean", "seed", "std", "version"]
    assert state["version"] == "v2"
    np.testing.assert_array_equal(state["context"].numpy(), split.observations[rows])
    np.testing.assert_array_equal(state["labels"].numpy(), split.data["actions"][rows])
    # The labels are part of the conditioning: other labels on the same rows give another embedding.
    relabelled = _LoggedSplit(split.observations, (split.data["actions"] + 1) % 3)
    assert not np.allclose(_fitted_tabpfn(relabelled, seed=3).transform(split.observations), out, atol=1e-2)


@pytest.mark.slow
def test_tabpfn_fit_is_seeded_and_deployment_matches_training():
    split = _logged_table()
    torch_state, numpy_state = torch.get_rng_state().clone(), np.random.get_state()[1].copy()
    first, second, other = _fitted_tabpfn(split, seed=0), _fitted_tabpfn(split, seed=0), _fitted_tabpfn(split, seed=1)
    assert torch.equal(torch.get_rng_state(), torch_state)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state)
    batch = first.transform(split.observations)
    np.testing.assert_array_equal(second.transform(split.observations), batch)
    assert not np.array_equal(other.fitted_state()["context"], first.fitted_state()["context"])
    assert not np.allclose(other.transform(split.observations), batch, atol=1e-2)
    # The simulator is embedded one observation per step; the dataset was embedded in one batch.
    single = np.concatenate([first.transform(split.observations[row : row + 1]) for row in range(20)])
    np.testing.assert_allclose(single, batch[:20], atol=1e-2)


@pytest.mark.slow
def test_tabpfn_state_round_trip_through_a_checkpoint_file(tmp_path):
    split = _logged_table()
    representation = _fitted_tabpfn(split)
    torch.save({"representation_state": representation.state_dict()}, tmp_path / "checkpoint.pt")
    state = torch.load(tmp_path / "checkpoint.pt", map_location="cpu")["representation_state"]
    restored = load_representation(TABPFN_CONFIG, state)
    # The context and the seed are in the checkpoint, so the conditioned model is rebuilt exactly.
    np.testing.assert_array_equal(restored.transform(split.observations), representation.transform(split.observations))


def _refuse_version(monkeypatch, refused: ModelVersion, error: Exception) -> list[ModelVersion]:
    """Makes the package fail with ``error`` for one version; returns the list of the versions it was asked for."""

    asked: list[ModelVersion] = []
    create = TabPFNClassifier.create_default_for_version

    def create_or_refuse(version, **settings):
        asked.append(version)
        if version == refused:
            raise error
        return create(version, **settings)

    monkeypatch.setattr(TabPFNClassifier, "create_default_for_version", create_or_refuse)
    return asked


@pytest.mark.slow
def test_tabpfn_falls_back_to_the_configured_version_only_when_the_license_is_missing(monkeypatch):
    split = _logged_table()
    expected = _fitted_tabpfn(split, seed=3).transform(split.observations)
    licensed = {**TABPFN_CONFIG, "tabpfn_version": "v3.5"}

    asked = _refuse_version(monkeypatch, ModelVersion.V3_5, TabPFNLicenseError())
    representation = build_representation(licensed, TABPFN_FEATURES)
    with pytest.warns(UserWarning, match=r"using tabpfn_fallback_version \(v2\) instead"):
        representation.fit(split, seed=3)
    assert asked == [ModelVersion.V3_5, ModelVersion.V2]
    # The fallback is the configured version on the same context: the embedding of a plain v2 run.
    np.testing.assert_array_equal(representation.transform(split.observations), expected)
    state = representation.state_dict()
    assert state["fitted"]["version"] == "v2"
    # Evaluation loads the version in the checkpoint; it does not ask for the refused one again.
    asked.clear()
    restored = load_representation(licensed, state)
    assert asked == [ModelVersion.V2]
    np.testing.assert_array_equal(restored.transform(split.observations), expected)

    # Any other failure is raised as it is: nothing else is hidden behind the fallback.
    _refuse_version(monkeypatch, ModelVersion.V3_5, RuntimeError("download failed"))
    with pytest.raises(RuntimeError, match="download failed"):
        build_representation(licensed, TABPFN_FEATURES).fit(split, seed=3)


def test_tabpfn_rejects_bad_settings_and_bad_contexts():
    split = _logged_table()
    with pytest.raises(ValueError, match="tabpfn_version"):
        build_representation({"representation": "tabpfn"}, TABPFN_FEATURES)
    with pytest.raises(ValueError, match="tabpfn_fallback_version"):
        build_representation({"representation": "tabpfn", "tabpfn_version": "v2"}, TABPFN_FEATURES)
    with pytest.raises(ValueError, match="not a valid ModelVersion"):
        build_representation({**TABPFN_CONFIG, "tabpfn_version": "v0"}, TABPFN_FEATURES)
    with pytest.raises(ValueError, match="tabpfn_context_size must be > 0"):
        build_representation({**TABPFN_CONFIG, "tabpfn_context_size": 0}, TABPFN_FEATURES)
    with pytest.raises(ValueError, match="exceeds the train split"):
        _fitted_tabpfn(split, tabpfn_context_size=121)
    # A context where every row has the same action has nothing to condition on.
    with pytest.raises(ValueError, match="at least two distinct logged actions"):
        _fitted_tabpfn(_LoggedSplit(split.observations, np.zeros(120)))


@pytest.mark.slow
def test_training_fits_tabpfn_on_the_train_split_only(tiny_datasets, tmp_path):
    root = tiny_datasets["simbank"]
    config = tiny_run_config("bc", "simbank", root, representation="tabpfn")
    result = train_offline(config)
    state = result.best_checkpoint["representation_state"]["fitted"]
    train = OfflineTransitionDataset(root / "train.npz")
    assert result.summary["representation_output_dim"] == 192
    assert state["context"].shape == (TINY_HYPERPARAMETERS["common"]["tabpfn_context_size"], train.obs_dim)
    # Every context row is a train row with its logged action.
    logged = {(*row, action) for row, action in zip(train.observations.tolist(), train.data["actions"].tolist())}
    assert {(*row, action) for row, action in zip(state["context"].tolist(), state["labels"].tolist())} <= logged
    # Replacing the validation split must not change the fitted state.
    collect_dataset(tiny_dataset_config("simbank", tmp_path, seed=99))
    swapped = {**config, "val_dataset_path": str(tmp_path / "val.npz")}
    other = train_offline(swapped).best_checkpoint["representation_state"]["fitted"]
    assert all(torch.equal(other[key], state[key]) for key in ("context", "labels", "mean", "std"))
