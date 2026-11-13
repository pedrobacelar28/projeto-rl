"""Layered configuration, the derived catalog, and experiment files."""

from __future__ import annotations

import json
import shutil

import pytest

from projeto_rl.algorithms import algorithm_names, get_algorithm
from projeto_rl.config import Benchmark, ConfigError, deep_merge, layer, load_experiment, parse_overrides
from projeto_rl.paths import CONFIG_ROOT


@pytest.fixture(scope="module")
def bench() -> Benchmark:
    return Benchmark.load()


def test_repository_configs_are_valid(bench):
    assert bench.validate() == []


def test_catalog_is_every_behavior_times_every_tier(bench):
    for env_id, spec in bench.dataset_specs.items():
        slug = bench.environments[env_id]["slug"]
        expected = {f"{slug}/{behavior}/{tier}" for behavior in spec["behaviors"] for tier in spec["tiers"]}
        assert expected == {entry.identifier for entry in bench.datasets if entry.env_id == env_id}
    assert len(bench.datasets) == 12


def test_every_registered_algorithm_runs_on_every_dataset(bench):
    for entry in bench.datasets:
        assert {method.algo for method in bench.methods(entry.identifier)} == set(algorithm_names())


def test_layers_apply_in_order_with_tier_sections(bench):
    dev = bench.run_config("industrial-benchmark/mediocre/dev", "dqn", artifact_root="/art")
    final = bench.run_config("industrial-benchmark/mediocre/final", "dqn", artifact_root="/art")
    assert dev["offline_train_steps"] == 10000 and dev["representation"] == "standardize"  # protocol
    assert (dev["eval_episodes"], final["eval_episodes"]) == (100, 1000)  # protocol tiers
    assert dev["tau"] == 0.01  # algorithm defaults
    assert dev["gamma"] == 0.99 and dev["env_kwargs"]["horizon"] == 250  # environment run settings
    iql_dev = bench.run_config("epicare/soc/dev", "iql", artifact_root="/art")
    iql_final = bench.run_config("epicare/soc/final", "iql", artifact_root="/art")
    assert (iql_dev["policy_hidden_dim"], iql_final["policy_hidden_dim"]) == (128, 256)  # algorithm tiers


def test_run_identity_paths_and_cli_overrides(bench):
    params = bench.run_config(
        "epicare/soc/dev",
        "iql",
        artifact_root="/art",
        seed=7,
        overrides=parse_overrides(['representation="standardize"', "env_kwargs.patient_modifiers=false"]),
    )
    assert (params["algo"], params["method_id"], params["seed"]) == ("iql", "iql", 7)
    assert params["dataset_path"] == "/art/datasets/epicare/soc/dev/train.npz"
    assert params["representation"] == "standardize" and params["env_kwargs"] == {"patient_modifiers": False}


def test_every_run_config_builds_its_algorithm(bench):
    for entry in bench.datasets:
        for method in bench.methods(entry.identifier):
            params = bench.run_config(entry.identifier, method.method_id, artifact_root="/art")
            assert get_algorithm(params["algo"]).name == method.algo


def test_dataset_config_carries_behavior_options(bench):
    config = bench.dataset_config("sales-promotion/ticket-band/final", artifact_root="/art")
    assert config["behavior_policy"] == "ticket_band" and config["behavior_epsilon"] == 0.1
    assert config["ticket_band_thresholds"] == [1.8, 12.0]
    assert (config["num_train_episodes"], config["num_val_episodes"]) == (7000, 1000)
    assert config["output_dir"] == "/art/datasets/sales-promotion/ticket-band/final"


def test_unknown_ids_are_reported(bench):
    with pytest.raises(ConfigError, match="Unknown dataset"):
        bench.dataset("epicare/nope/dev")
    with pytest.raises(ConfigError, match="Available"):
        bench.method("epicare/soc/dev", "ppo")


def test_validation_catches_broken_configs(tmp_path):
    root = tmp_path / "configs"
    shutil.copytree(CONFIG_ROOT, root)
    (root / "algorithms" / "cql.json").unlink()
    spec = json.loads((root / "datasets" / "epicare.json").read_text())
    spec["behaviors"]["soc"]["behavior_policy"] = "unknown"
    spec["behaviors"]["soc"].pop("behavior_epsilon")
    (root / "datasets" / "epicare.json").write_text(json.dumps(spec))
    env = json.loads((root / "envs" / "simbank.json").read_text())
    env["feature_mix"] = "mixed"
    (root / "envs" / "simbank.json").write_text(json.dumps(env))
    protocol = json.loads((root / "protocol.json").read_text())
    protocol["representation"] = "pca"
    (root / "protocol.json").write_text(json.dumps(protocol))
    errors = Benchmark.load(root).validate()
    assert any("configs/algorithms/cql.json" in error for error in errors)
    assert any("undeclared behavior policy" in error for error in errors)
    assert any("normalization.high" in error for error in errors)
    assert any("behavior_epsilon" in error for error in errors)
    assert any("feature_mix" in error for error in errors)
    assert any("protocol.json representation" in error for error in errors)


def test_experiments_load_and_reject_unknown_keys(tmp_path):
    final = load_experiment("configs/experiments/final.json")
    assert final["seeds"] == [0, 1, 2] and final["representations"] == [
        "raw",
        "standardize",
        "spr",
        "tjepa",
        "tabpfn",
        "tdjepa",
    ]
    assert final["reference_representation"] == "standardize"
    assert len(final["datasets"]) * len(final["methods"]) * len(final["representations"]) * len(final["seeds"]) == 432
    # wide.json is final.json restricted to the two wide environments.
    wide = load_experiment("configs/experiments/wide.json")
    assert wide["datasets"] == ["citylearn/time-of-use/final", "grid2op/rule/final"]
    assert {key: wide[key] for key in ("methods", "representations", "reference_representation", "seeds")} == {
        key: final[key] for key in ("methods", "representations", "reference_representation", "seeds")
    }
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**json.loads((CONFIG_ROOT / "experiments" / "dev.json").read_text()), "tier": "x"}))
    with pytest.raises(ConfigError, match="unknown keys"):
        load_experiment(bad)
    bad.write_text(
        json.dumps({**json.loads((CONFIG_ROOT / "experiments" / "dev.json").read_text()), "representations": ["pca"]})
    )
    with pytest.raises(ConfigError, match="unregistered"):
        load_experiment(bad)
    bad.write_text(
        json.dumps(
            {**json.loads((CONFIG_ROOT / "experiments" / "dev.json").read_text()), "reference_representation": "raw2"}
        )
    )
    with pytest.raises(ConfigError, match="reference_representation"):
        load_experiment(bad)


def test_layer_and_merge_helpers():
    assert layer({"a": 1, "description": "x", "tiers": {"dev": {"a": 2, "b": {"c": 3}}}}, "dev") == {
        "a": 2,
        "b": {"c": 3},
    }
    assert deep_merge({"env_kwargs": {"x": 1, "y": 2}}, {"env_kwargs": {"y": 3}}) == {"env_kwargs": {"x": 1, "y": 3}}
    with pytest.raises(ConfigError):
        parse_overrides(["no_equals_sign"])
