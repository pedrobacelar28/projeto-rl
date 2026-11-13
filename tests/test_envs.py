"""Contract tests run against every registered environment."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import TINY_DATASETS, tiny_dataset_config

from projeto_rl.data.collect import behavior_policy, collect_dataset
from projeto_rl.data.dataset import REQUIRED_ARRAYS, OfflineTransitionDataset
from projeto_rl.envs import FeatureSpec, environment_ids, make_env
from projeto_rl.evaluation import evaluate_policy

ENVIRONMENTS = environment_ids()


def _env(env_id: str, purpose: str = "evaluate"):
    return make_env(TINY_DATASETS[env_id], seed=0, purpose=purpose)


def test_every_environment_has_tiny_fixtures():
    assert set(ENVIRONMENTS) == set(TINY_DATASETS)


def test_make_env_requires_a_known_purpose():
    with pytest.raises(ValueError, match="purpose"):
        make_env(TINY_DATASETS["epicare"], purpose="train")


def test_feature_spec_validation():
    with pytest.raises(ValueError, match="kind"):
        FeatureSpec("x", "text")
    with pytest.raises(ValueError, match="cardinality"):
        FeatureSpec("x", "categorical")
    with pytest.raises(ValueError, match="takes no cardinality"):
        FeatureSpec("x", "numeric", 3)
    FeatureSpec("x", "ordinal", 3).check(np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="codes"):
        FeatureSpec("x", "ordinal", 3).check(np.array([0.0, 3.0]))
    with pytest.raises(ValueError, match="codes"):
        FeatureSpec("x", "cyclic", 7).check(np.array([1.5]))
    with pytest.raises(ValueError, match="Binary"):
        FeatureSpec("x", "binary").check(np.array([0.0, 0.5]))
    with pytest.raises(ValueError, match="non-finite"):
        FeatureSpec("x", "numeric").check(np.array([np.nan]))


@pytest.mark.parametrize("env_id", ENVIRONMENTS)
def test_spec_matches_observations_and_info(env_id):
    env = _env(env_id)
    try:
        obs, info = env.reset(seed=0)
        spec = env.spec
        assert obs.dtype == np.float32 and obs.shape == (spec.obs_dim,)
        assert len(spec.features) == spec.obs_dim == len(spec.feature_names)
        assert len(spec.action_labels) == spec.num_actions
        assert info["action_mask"].shape == (spec.num_actions,)
        assert set(np.flatnonzero(info["action_mask"])) == set(info["admissible_actions"])
        assert isinstance(info["state_id"], int)
        assert list(env.transition_metrics(info)) == list(spec.metric_names)
        observations = [obs]
        for _ in range(15):
            obs, reward, terminated, truncated, info = env.step(int(info["admissible_actions"][-1]))
            assert np.isfinite(reward) and isinstance(terminated, bool) and isinstance(truncated, bool)
            assert info["action_mask"].shape == (spec.num_actions,)
            observations.append(obs)
            if terminated or truncated:
                break
        observations = np.stack(observations)
        for index, feature in enumerate(spec.features):
            feature.check(observations[:, index])
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENVIRONMENTS)
def test_reset_seed_reproduces_trajectory(env_id):
    env = _env(env_id)
    try:
        rollouts = []
        for _ in range(2):
            obs, info = env.reset(seed=21)
            steps = [obs]
            for _ in range(5):
                obs, reward, terminated, truncated, info = env.step(int(info["admissible_actions"][-1]))
                steps.append(np.append(obs, reward))
                if terminated or truncated:
                    break
            rollouts.append(steps)
        assert len(rollouts[0]) == len(rollouts[1])
        for first, second in zip(*rollouts):
            np.testing.assert_array_equal(first, second)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENVIRONMENTS)
def test_declared_policies_are_valid_distributions(env_id):
    env = _env(env_id)
    try:
        for name in env.behavior_policy_ids:
            policy = env.make_policy(name, seed=0, options=TINY_DATASETS[env_id])
            obs, info = env.reset(seed=1)
            policy.reset_episode()
            for deterministic in (False, True):
                probabilities = policy.action_probabilities(obs, info, deterministic=deterministic)
                assert probabilities.shape == (env.num_actions,), name
                assert np.isclose(probabilities.sum(), 1.0), name
                assert np.all(probabilities[info["action_mask"] <= 0] == 0.0), name
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENVIRONMENTS)
def test_behavior_policy_evaluates_stochastically(env_id):
    config = {**TINY_DATASETS[env_id], "seed": 0, "method_id": "baseline"}
    env = _env(env_id)
    try:
        policy, _ = behavior_policy(config, env)
    finally:
        env.close()
    summary = evaluate_policy(
        policy, config=config, num_episodes=2, eval_seed_base=0, export_csv=False, deterministic=False
    )["summary"]
    assert summary["episodes"] == 2 and np.isfinite(summary["mean_return"])


@pytest.mark.parametrize("env_id", ENVIRONMENTS)
def test_collected_dataset_schema_features_and_splits(env_id, tiny_datasets):
    root = tiny_datasets[env_id]
    splits = {name: np.load(root / f"{name}.npz") for name in ("train", "val")}
    for name, data in splits.items():
        assert REQUIRED_ARRAYS.issubset(data.files), name
        assert data["observations"].dtype == np.float32
        assert np.isfinite(data["behavior_logprobs"]).all() and (data["behavior_logprobs"] <= 0).all()
        assert all(mask[action] == 1 for action, mask in zip(data["actions"], data["action_masks"]))
    assert set(splits["train"]["episode_ids"].tolist()).isdisjoint(splits["val"]["episode_ids"].tolist())
    dataset = OfflineTransitionDataset(root / "train.npz")
    env = _env(env_id, purpose="collect")
    assert dataset.features == env.spec.features
    assert dataset.metadata["behavior"]["epsilon"] == TINY_DATASETS[env_id]["behavior_epsilon"]
    for index, feature in enumerate(dataset.features):
        feature.check(dataset.observations[:, index])


def _kept_episodes(path, fraction: float, seed: int) -> OfflineTransitionDataset:
    dataset = OfflineTransitionDataset(path)
    dataset.keep_episode_fraction(fraction, seed=seed)
    return dataset


def test_an_episode_fraction_keeps_whole_episodes_of_a_seeded_shuffle(tiny_datasets):
    path = tiny_datasets["simbank"] / "train.npz"
    full = OfflineTransitionDataset(path)
    episodes = np.unique(full.data["episode_ids"])
    assert len(episodes) == 6
    # The rule, redone here: the first episodes of a shuffle that depends on the seed and the count only.
    order = np.random.default_rng(7).permutation(6)
    kept_ids = []
    for fraction, count in ((1 / 3, 2), (2 / 3, 4), (1.0, 6)):
        dataset = _kept_episodes(path, fraction, seed=7)
        rows = np.isin(full.data["episode_ids"], episodes[order[:count]])
        assert dataset.num_episodes == count and len(dataset) == rows.sum()
        # Every array keeps exactly the rows of those episodes, in storage order.
        for key, values in full.data.items():
            np.testing.assert_array_equal(dataset.data[key], values[rows], err_msg=key)
        # The episode index is rebuilt: one episode per entry, in timestep order, covering every row once.
        for indices in dataset.episode_indices:
            assert len(set(dataset.data["episode_ids"][indices].tolist())) == 1
            assert (np.diff(dataset.data["timesteps"][indices]) > 0).all()
        assert sorted(np.concatenate(dataset.episode_indices).tolist()) == list(range(len(dataset)))
        kept_ids.append(set(dataset.data["episode_ids"].tolist()))
    # A smaller fraction is a subset of a larger one.
    assert kept_ids[0] < kept_ids[1] < kept_ids[2] == set(episodes.tolist())
    assert len({frozenset(_kept_episodes(path, 1 / 3, seed).data["episode_ids"].tolist()) for seed in range(6)}) > 1
    for fraction in (0.0, 1.5):
        with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
            full.keep_episode_fraction(fraction, seed=0)
    with pytest.raises(ValueError, match="keeps none of its 6 episodes"):
        full.keep_episode_fraction(0.01, seed=0)


@pytest.mark.parametrize("env_id", ENVIRONMENTS)
def test_dataset_collection_is_deterministic(env_id, tmp_path):
    for name in ("a", "b"):
        collect_dataset(tiny_dataset_config(env_id, tmp_path / name, seed=13))
    for split in ("train.npz", "val.npz"):
        first, second = np.load(tmp_path / "a" / split), np.load(tmp_path / "b" / split)
        for key in first.files:
            np.testing.assert_array_equal(first[key], second[key], err_msg=f"{split}:{key}")


# How each tiny environment ends its episodes: Industrial Benchmark only by its
# time limit, SalesPromotion and SimBank only by a true terminal state, EpiCare
# by remission or an adverse event (terminal) or by its visit limit (timeout).
EPISODE_ENDS = {
    "industrial_benchmark": {"timeouts"},
    "sales_promotion": {"terminals"},
    "simbank": {"terminals"},
    "epicare": {"terminals", "timeouts"},
    "citylearn": {"timeouts"},
    # A blackout is terminal; surviving the day is a time limit.
    "grid2op": {"terminals", "timeouts"},
}


@pytest.mark.parametrize("env_id", ENVIRONMENTS)
def test_time_limits_are_logged_as_timeouts_not_terminals(env_id, tiny_datasets):
    data = np.load(tiny_datasets[env_id] / "train.npz")
    terminals, timeouts = data["terminals"], data["timeouts"]
    assert not (terminals & timeouts).any()
    seen = set()
    for episode in np.unique(data["episode_ids"]):
        rows = np.flatnonzero(data["episode_ids"] == episode)
        rows = rows[np.argsort(data["timesteps"][rows])]
        last = rows[-1]
        # Exactly the last transition of every episode ends it.
        assert not terminals[rows[:-1]].any() and not timeouts[rows[:-1]].any()
        assert terminals[last] != timeouts[last]
        seen.add("terminals" if terminals[last] else "timeouts")
        if env_id in ("industrial_benchmark", "citylearn"):
            assert len(rows) == TINY_DATASETS[env_id]["env_kwargs"]["horizon"]
        if env_id == "epicare" and timeouts[last]:
            assert len(rows) == 8  # EpiCare's max_visits
    assert seen <= EPISODE_ENDS[env_id]


def test_dataset_of_an_older_schema_is_stale(tiny_datasets, tmp_path):
    import shutil

    from projeto_rl.cli.common import dataset_status
    from projeto_rl.config import dataset_paths
    from projeto_rl.utils.io import load_json, save_json

    root = dataset_paths("epicare/soc/dev", tmp_path)["root"]
    shutil.copytree(tiny_datasets["epicare"], root)
    assert dataset_status("epicare/soc/dev", tmp_path)["ready"]
    metadata = load_json(root / "metadata.json")
    save_json({**metadata, "schema_version": 4}, root / "metadata.json")
    status = dataset_status("epicare/soc/dev", tmp_path)
    assert status["stale"] and not status["ready"]
    with pytest.raises(ValueError, match="schema 4"):
        collect_dataset({**tiny_dataset_config("epicare", root), "overwrite": False})
