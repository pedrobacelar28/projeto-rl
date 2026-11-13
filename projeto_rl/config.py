"""Benchmark configuration: environments, datasets, and layered run configs.

``configs/`` holds five kinds of files, each small and flat:

``protocol.json``                offline protocol shared by every run
``algorithms/<algo>.json``       default hyperparameters of one algorithm
``envs/<env_id>.json``           environment identity, run settings, and score normalization
``datasets/<env_id>.json``       behavior policy x size tiers of one environment
``experiments/<name>.json``      datasets x methods x representations x seeds of one experiment,
                                 the reference representation the others are compared to, and
                                 optionally the share of the episodes the methods train on

A run config is resolved as protocol -> algorithm -> environment run
settings -> CLI ``--set``. Within any layer, a ``tiers`` object holds values
that apply only to the dataset's tier. Datasets and compatible methods are
derived from these files and the algorithm registry; there is no
hand-maintained list of (dataset, method) pairs.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from projeto_rl.algorithms import algorithms
from projeto_rl.data.collect import SCHEMA_VERSION
from projeto_rl.envs import environment_ids, get_environment
from projeto_rl.paths import CONFIG_ROOT, PROJECT_ROOT
from projeto_rl.representations import representation_names

LAYER_METADATA = {"description", "tiers"}
# Summaries report homogeneous (one column type) and heterogeneous tables apart.
FEATURE_MIXES = ("homogeneous", "heterogeneous")
TIERS = ("dev", "final")


class ConfigError(ValueError):
    """Raised for invalid configuration files or unknown dataset/method ids."""


@dataclass(frozen=True)
class DatasetEntry:
    identifier: str
    env_id: str
    behavior: str
    tier: str
    behavior_policy: str
    description: str
    train_episodes: int


@dataclass(frozen=True)
class MethodEntry:
    method_id: str
    algo: str
    description: str


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def layer(payload: dict[str, Any], tier: str) -> dict[str, Any]:
    """Parameters of one config layer for ``tier``."""

    params = {key: value for key, value in payload.items() if key not in LAYER_METADATA}
    return deep_merge(params, payload.get("tiers", {}).get(tier, {}))


def parse_overrides(raw: list[str] | None) -> dict[str, Any]:
    """Parse ``key=json-value`` CLI overrides; dotted keys address nested objects."""

    parsed: dict[str, Any] = {}
    for item in raw or []:
        if "=" not in item:
            raise ConfigError(f"Override must use key=json-value, got {item!r}")
        key, raw_value = item.split("=", 1)
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Override {key!r} must contain a JSON value: {exc.msg}") from exc
        parts = key.split(".")
        if not all(parts):
            raise ConfigError(f"Invalid override key {key!r}")
        target = parsed
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return parsed


def dataset_paths(dataset_id: str, artifact_root: str | Path) -> dict[str, Path]:
    root = Path(artifact_root) / "datasets" / dataset_id
    return {
        "root": root,
        "train": root / "train.npz",
        "validation": root / "val.npz",
        "metadata": root / "metadata.json",
    }


class Benchmark:
    def __init__(self, root: str | Path = CONFIG_ROOT):
        self.root = Path(root)
        self.protocol = _read(self.root / "protocol.json")
        self.environments = {path.stem: _read(path) for path in sorted((self.root / "envs").glob("*.json"))}
        self.dataset_specs = {path.stem: _read(path) for path in sorted((self.root / "datasets").glob("*.json"))}
        self._datasets: dict[str, DatasetEntry] = {}
        for env_id, spec in self.dataset_specs.items():
            slug = self.environments.get(env_id, {}).get("slug", env_id)
            for behavior, options in spec.get("behaviors", {}).items():
                for tier, counts in spec.get("tiers", {}).items():
                    identifier = f"{slug}/{behavior}/{tier}"
                    self._datasets[identifier] = DatasetEntry(
                        identifier=identifier,
                        env_id=env_id,
                        behavior=behavior,
                        tier=tier,
                        behavior_policy=str(options["behavior_policy"]),
                        description=str(options.get("description", "")),
                        train_episodes=int(counts["num_train_episodes"]),
                    )

    @classmethod
    def load(cls, root: str | Path | None = None) -> "Benchmark":
        return cls(root or CONFIG_ROOT)

    # -- discovery ------------------------------------------------------
    @property
    def datasets(self) -> list[DatasetEntry]:
        return sorted(self._datasets.values(), key=lambda item: item.identifier)

    def dataset(self, identifier: str) -> DatasetEntry:
        try:
            return self._datasets[identifier]
        except KeyError as exc:
            raise ConfigError(
                f"Unknown dataset {identifier!r}. Available: {', '.join(sorted(self._datasets))}"
            ) from exc

    def methods(self, dataset_id: str | None = None) -> list[MethodEntry]:
        action_type = None if dataset_id is None else self.environments[self.dataset(dataset_id).env_id]["action_type"]
        return [
            MethodEntry(method_id=cls.method_id, algo=cls.name, description=cls.description)
            for cls in algorithms()
            if action_type is None or action_type in cls.action_types
        ]

    def method(self, dataset_id: str, method_id: str) -> MethodEntry:
        compatible = self.methods(dataset_id)
        for entry in compatible:
            if method_id in {entry.method_id, entry.algo}:
                return entry
        choices = ", ".join(entry.method_id for entry in compatible)
        raise ConfigError(f"Method {method_id!r} is not available for {dataset_id!r}. Available: {choices}")

    # -- resolution -----------------------------------------------------
    def _environment_run(self, env_id: str) -> dict[str, Any]:
        return {"env_id": env_id, **self.environments[env_id].get("run", {})}

    def dataset_config(self, dataset_id: str, *, artifact_root: str | Path, force: bool = False) -> dict[str, Any]:
        entry = self.dataset(dataset_id)
        spec = self.dataset_specs[entry.env_id]
        options = {key: value for key, value in spec["behaviors"][entry.behavior].items() if key != "description"}
        return {
            **self._environment_run(entry.env_id),
            "dataset_id": entry.identifier,
            "seed": int(spec["seed"]),
            **spec["tiers"][entry.tier],
            **options,
            "output_dir": str(dataset_paths(entry.identifier, artifact_root)["root"]),
            "overwrite": bool(force),
        }

    def run_config(
        self,
        dataset_id: str,
        method_id: str,
        *,
        artifact_root: str | Path,
        seed: int | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        dataset = self.dataset(dataset_id)
        method = self.method(dataset_id, method_id)
        tier = dataset.tier
        config = layer(self.protocol, tier)
        config = deep_merge(config, layer(self._algorithm_file(method.algo), tier))
        config = deep_merge(config, self._environment_run(dataset.env_id))
        paths = dataset_paths(dataset.identifier, artifact_root)
        config.update(
            {
                "algo": method.algo,
                "method_id": method.method_id,
                "dataset_id": dataset.identifier,
                "seed": int(0 if seed is None else seed),
                "dataset_path": str(paths["train"]),
                "val_dataset_path": str(paths["validation"]),
                # Part of the run identity: a run trained on an older dataset
                # schema is never reused after the datasets are rebuilt.
                "dataset_schema_version": SCHEMA_VERSION,
            }
        )
        return deep_merge(config, overrides or {})

    def _algorithm_file(self, algo: str) -> dict[str, Any]:
        path = self.root / "algorithms" / f"{algo}.json"
        if not path.exists():
            raise ConfigError(f"Missing default hyperparameters for algorithm {algo!r}: {path}")
        return _read(path)

    # -- validation -----------------------------------------------------
    def validate(self) -> list[str]:
        errors: list[str] = []
        registered = set(environment_ids())
        for env_id in sorted(set(self.environments) | set(self.dataset_specs)):
            if env_id not in registered:
                errors.append(f"configs reference unregistered environment {env_id!r}")
            if env_id not in self.environments:
                errors.append(f"missing configs/envs/{env_id}.json")
            if env_id not in self.dataset_specs:
                errors.append(f"missing configs/datasets/{env_id}.json")
        for env_id, spec in self.dataset_specs.items():
            if env_id not in registered:
                continue
            declared = set(get_environment(env_id).behavior_policy_ids)
            for behavior, options in spec.get("behaviors", {}).items():
                if options.get("behavior_policy") not in declared:
                    errors.append(
                        f"dataset {env_id}/{behavior} uses undeclared behavior policy {options.get('behavior_policy')!r}"
                    )
            for tier in spec.get("tiers", {}):
                if tier not in TIERS:
                    errors.append(f"configs/datasets/{env_id}.json has unknown tier {tier!r}")
            if self.environments.get(env_id, {}).get("feature_mix") not in FEATURE_MIXES:
                errors.append(f"configs/envs/{env_id}.json feature_mix must be one of {FEATURE_MIXES}")
            normalization = self.environments.get(env_id, {}).get("normalization", {})
            behaviors = {options.get("behavior_policy") for options in spec.get("behaviors", {}).values()}
            if normalization.get("low") not in declared:
                errors.append(f"configs/envs/{env_id}.json normalization.low is not a declared policy")
            if behaviors != {normalization.get("high")}:
                errors.append(
                    f"configs/envs/{env_id}.json normalization.high must be the dataset behavior policy {sorted(behaviors)}"
                )
            for behavior, options in spec.get("behaviors", {}).items():
                epsilon = options.get("behavior_epsilon")
                if not isinstance(epsilon, (int, float)) or not 0.0 <= float(epsilon) <= 1.0:
                    errors.append(f"dataset {env_id}/{behavior} needs behavior_epsilon in [0, 1]")
        for cls in algorithms():
            if not (self.root / "algorithms" / f"{cls.name}.json").exists():
                errors.append(f"missing configs/algorithms/{cls.name}.json")
        if self.protocol.get("representation") not in representation_names():
            errors.append(f"protocol.json representation must be one of {representation_names()}")
        return errors


EXPERIMENT_KEYS = {
    "description",
    "datasets",
    "methods",
    "representations",
    "reference_representation",
    "seeds",
    "progress_interval_seconds",
}
# Optional: the share of the train (and validation) episodes the methods see.
# An experiment without it trains on every episode and its run configs do not
# carry the key, so their identity is the one they always had.
OPTIONAL_EXPERIMENT_KEYS = {"train_episode_fraction"}


def load_experiment(path: str | Path) -> dict[str, Any]:
    """Read ``configs/experiments/<name>.json``: datasets x methods x representations x seeds."""

    path = Path(path)
    path = path if path.is_absolute() else PROJECT_ROOT / path
    payload = _read(path)
    unknown = sorted(set(payload) - EXPERIMENT_KEYS - OPTIONAL_EXPERIMENT_KEYS)
    missing = sorted(EXPERIMENT_KEYS - {"description"} - set(payload))
    if unknown or missing:
        raise ConfigError(f"Experiment {path} has unknown keys {unknown} or misses {missing}")
    unregistered = sorted(set(payload["representations"]) - set(representation_names()))
    if unregistered:
        raise ConfigError(f"Experiment {path} uses unregistered representations {unregistered}")
    if payload["reference_representation"] not in payload["representations"]:
        raise ConfigError(
            f"Experiment {path}: reference_representation {payload['reference_representation']!r} "
            f"is not one of its representations {payload['representations']}"
        )
    fraction = payload.get("train_episode_fraction")
    if fraction is not None and not (isinstance(fraction, (int, float)) and 0.0 < fraction < 1.0):
        raise ConfigError(f"Experiment {path}: train_episode_fraction must be a number in (0, 1), got {fraction!r}")
    return {
        "path": path,
        "description": str(payload.get("description", "")),
        "datasets": [str(item) for item in payload["datasets"]],
        "methods": [str(item) for item in payload["methods"]],
        "representations": [str(item) for item in payload["representations"]],
        "reference_representation": str(payload["reference_representation"]),
        "seeds": [int(item) for item in payload["seeds"]],
        "progress_interval_seconds": float(payload["progress_interval_seconds"]),
        "train_episode_fraction": None if fraction is None else float(fraction),
    }
