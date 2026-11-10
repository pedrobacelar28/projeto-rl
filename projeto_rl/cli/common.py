"""Helpers shared by the CLI command modules."""

from __future__ import annotations

import argparse
import contextlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from projeto_rl.algorithms import get_algorithm
from projeto_rl.cli.console import CliConsole
from projeto_rl.config import Benchmark, ConfigError, dataset_paths, parse_overrides
from projeto_rl.data.collect import SCHEMA_VERSION
from projeto_rl.operations.artifacts import artifact_root, initial_manifest, run_directory
from projeto_rl.utils.io import load_json


def benchmark(args: argparse.Namespace) -> Benchmark:
    loaded = Benchmark.load(args.config_root)
    errors = loaded.validate()
    if errors:
        raise ConfigError("Configuration validation failed:\n- " + "\n- ".join(errors))
    return loaded


def root(args: argparse.Namespace) -> Path:
    return artifact_root(args.artifact_root)


def console(args: argparse.Namespace) -> CliConsole:
    return CliConsole(json_mode=bool(args.json), color=args.color)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def compact_count(value: int | float) -> str:
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if value >= 1_000:
        return f"{value / 1_000:.3g}k"
    return str(value)


def parse_seeds(raw: str) -> list[int]:
    try:
        seeds = [int(item) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--seeds must contain comma-separated integers") from exc
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")
    return seeds


@contextlib.contextmanager
def device(requested: str) -> Iterator[None]:
    previous = os.environ.get("PROJETO_RL_DEVICE")
    os.environ["PROJETO_RL_DEVICE"] = requested
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PROJETO_RL_DEVICE", None)
        else:
            os.environ["PROJETO_RL_DEVICE"] = previous


def dataset_status(dataset_id: str, artifacts: Path) -> dict[str, Any]:
    paths = dataset_paths(dataset_id, artifacts)
    required = {"train": paths["train"], "validation": paths["validation"], "metadata": paths["metadata"]}
    missing = {name: str(path) for name, path in required.items() if not path.exists()}
    schema = None if missing else load_json(paths["metadata"]).get("schema_version")
    # A dataset of an older schema is not usable: it lacks arrays the loader requires.
    stale = not missing and schema != SCHEMA_VERSION
    return {
        "ready": not missing and not stale,
        "paths": {name: str(path) for name, path in paths.items()},
        "missing": missing,
        "schema_version": schema,
        "stale": stale,
    }


def dataset_payload(bench: Benchmark, artifacts: Path, dataset_id: str) -> dict[str, Any]:
    entry = bench.dataset(dataset_id)
    status = dataset_status(dataset_id, artifacts)
    payload: dict[str, Any] = {
        "id": entry.identifier,
        "environment": entry.env_id,
        "behavior": entry.behavior,
        "behavior_policy": entry.behavior_policy,
        "tier": entry.tier,
        "train_episodes": entry.train_episodes,
        "description": entry.description,
        "availability": status,
        "methods": [method.method_id for method in bench.methods(dataset_id)],
    }
    metadata_path = Path(status["paths"]["metadata"])
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        payload["schema_version"] = metadata.get("schema_version")
        payload["environment_spec"] = metadata.get("environment")
        payload["splits"] = {key: metadata[key] for key in ("train", "validation") if key in metadata}
    return payload


def data_preflight(params: dict[str, Any]) -> dict[str, Any]:
    required = {"train": Path(params["dataset_path"]), "validation": Path(params["val_dataset_path"])}
    missing = {name: str(path) for name, path in required.items() if not path.exists()}
    return {"ready": not missing, "missing": missing, "paths": {name: str(path) for name, path in required.items()}}


def prepare_run(
    args: argparse.Namespace, *, dataset_id: str, method_id: str, seed: int | None, representation: str | None = None
) -> dict[str, Any]:
    """Resolve one run: params, output directory, manifest, and data preflight."""

    bench = benchmark(args)
    overrides = parse_overrides(args.overrides)
    if representation is not None:
        overrides["representation"] = representation
    params = bench.run_config(dataset_id, method_id, artifact_root=root(args), seed=seed, overrides=overrides)
    params["show_progress"] = not args.no_progress
    output_dir = run_directory(params, root=root(args))
    return {
        "params": params,
        "output_dir": output_dir,
        "manifest": initial_manifest(params, run_dir=output_dir, device=args.device),
        "preflight": data_preflight(params),
    }


def run_plan(run: dict[str, Any]) -> dict[str, Any]:
    params = run["params"]
    cls = get_algorithm(params["algo"])
    patience = int(params.get("early_stopping_patience", 20))
    return {
        "run_id": run["manifest"]["run_id"],
        "output_dir": str(run["output_dir"]),
        "dataset_id": params["dataset_id"],
        "environment": params["env_id"],
        "method_id": params["method_id"],
        "representation": params["representation"],
        "seed": params["seed"],
        "budget": {
            "optimization_steps": int(params["offline_train_steps"]),
            "batch_size": int(params["offline_batch_size"]),
            "sampled_transitions_estimate": int(params["offline_train_steps"]) * int(params["offline_batch_size"]),
            "final_eval_episodes": int(params["eval_episodes"]),
        },
        "model_selection": {
            "monitor": "validation_loss (minimize)",
            "meaning": cls.validation_objective,
            "check_every_steps": int(params["offline_log_interval"]),
            "early_stopping_patience": patience,
            "checkpoint": "lowest validation_loss",
        },
        "preflight": run["preflight"],
    }


def add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="Typed config override for this run only.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--force", action="store_true", help="Run even when a completed artifact already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and display work without writing artifacts.")
    parser.add_argument("--no-progress", action="store_true")


def add_queue_options(parser: argparse.ArgumentParser, *, default_seeds: str | None) -> None:
    """Queue flags; ``default_seeds=None`` when the seeds come from an experiment file."""

    if default_seeds is not None:
        parser.add_argument("--seeds", default=default_seeds)
    parser.add_argument("--execute", action="store_true", help="Run the jobs; without it the plan is only printed.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--gpus")
    parser.add_argument("--follow", action="store_true", help="Show live job progress while the queue runs.")
