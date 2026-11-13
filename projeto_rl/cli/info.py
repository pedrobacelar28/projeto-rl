"""`doctor`, `catalog`, and `validate`."""

from __future__ import annotations

import importlib
import sys

import torch

from projeto_rl.algorithms import algorithms
from projeto_rl.cli import common
from projeto_rl.config import Benchmark
from projeto_rl.envs import environment_ids, make_env
from projeto_rl.paths import PROJECT_ROOT


def doctor(args) -> int:
    console = common.console(args)
    packages = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "torch",
        "gymnasium",
        "rich",
        "simpy",
        "pm4py",
        "epicare",
        "tabpfn",
        "citylearn",
        "grid2op",
        "lightsim2grid",
    ):
        try:
            packages[name] = getattr(importlib.import_module(name), "__version__", "available")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            packages[name] = f"unavailable: {exc}"
    payload = {
        "project_root": str(PROJECT_ROOT),
        "artifact_root": str(common.root(args)),
        "python": sys.version.split()[0],
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
        "environments": environment_ids(),
        "algorithms": [cls.method_id for cls in algorithms()],
        "config_errors": Benchmark.load(args.config_root).validate(),
    }
    if args.json:
        console.emit(payload)
        return 0
    console.title("projeto-rl doctor", "Runtime, hardware, and registered components")
    console.table(
        "Runtime",
        ("item", "value"),
        [
            ("Python", payload["python"]),
            ("Artifact root", payload["artifact_root"]),
            ("CUDA", payload["cuda_available"]),
            ("GPUs", ", ".join(payload["cuda_devices"]) or "none"),
            ("Environments", ", ".join(payload["environments"])),
            ("Algorithms", ", ".join(payload["algorithms"])),
        ],
    )
    console.table("Packages", ("package", "version"), packages.items())
    if payload["config_errors"]:
        console.warning("Configuration errors:\n- " + "\n- ".join(payload["config_errors"]))
    return 0


def catalog(args) -> int:
    bench = common.benchmark(args)
    console = common.console(args)
    datasets = [common.dataset_payload(bench, common.root(args), entry.identifier) for entry in bench.datasets]
    methods = [method.__dict__ for method in bench.methods()]
    if args.json:
        console.emit({"datasets": datasets, "methods": methods})
        return 0
    console.title("projeto-rl catalog", "Datasets (environment/behavior/tier) and the methods that run on them")
    console.table(
        "Datasets",
        ("id", "behavior policy", "train episodes", "state"),
        [
            (
                item["id"],
                item["behavior_policy"],
                common.compact_count(item["train_episodes"]),
                "ready" if item["availability"]["ready"] else "build required",
            )
            for item in datasets
        ],
    )
    console.table(
        "Methods",
        ("method", "description"),
        [(item["method_id"], item["description"]) for item in methods],
    )
    return 0


def validate(args) -> int:
    bench = common.benchmark(args)
    console = common.console(args)
    payload = {"config": "valid", "datasets": len(bench.datasets), "methods": len(bench.methods()), "smoke": []}
    if args.smoke:
        for env_id in environment_ids():
            env = make_env({"env_id": env_id, **bench.environments[env_id].get("run", {})}, seed=0, purpose="evaluate")
            try:
                _, info = env.reset(seed=0)
                _, reward, terminated, truncated, _ = env.step(int(info["admissible_actions"][0]))
                payload["smoke"].append(
                    {"env_id": env_id, "reward": float(reward), "done": bool(terminated or truncated)}
                )
            finally:
                env.close()
    if args.json:
        console.emit(payload)
    else:
        console.success(f"Configuration is valid: {payload['datasets']} datasets, {payload['methods']} methods")
        if payload["smoke"]:
            console.table(
                "Environment smoke test",
                ("env", "reward", "done"),
                [(item["env_id"], item["reward"], item["done"]) for item in payload["smoke"]],
            )
    return 0


def register(subparsers) -> None:
    subparsers.add_parser("doctor", help="Inspect dependencies, hardware, and registered components.").set_defaults(
        handler=doctor
    )
    subparsers.add_parser("catalog", help="List datasets and methods.").set_defaults(handler=catalog)
    parser = subparsers.add_parser("validate", help="Validate configs and optionally smoke-test each environment.")
    parser.add_argument("--smoke", action="store_true")
    parser.set_defaults(handler=validate)
