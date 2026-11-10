"""Reference returns that normalize scores: the random policy and the dataset's behavior policy.

Both are evaluated with the same evaluation seeds and episode count as the
trained runs of the dataset's tier, so a normalized score of 0 means "as good
as random" and 100 means "as good as the policy that logged the data".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from projeto_rl.config import Benchmark, layer
from projeto_rl.data.collect import behavior_policy
from projeto_rl.envs import make_env
from projeto_rl.evaluation import evaluate_policy
from projeto_rl.utils.io import load_json, save_json

SCHEMA_VERSION = 1


def baseline_path(dataset_id: str, artifact_root: str | Path) -> Path:
    return Path(artifact_root) / "reports" / "baselines" / dataset_id / "baseline_summary.json"


def baseline_config(bench: Benchmark, dataset_id: str, *, artifact_root: str | Path) -> dict[str, Any]:
    """Dataset collection settings plus the evaluation protocol of the dataset's tier."""

    entry = bench.dataset(dataset_id)
    protocol = layer(bench.protocol, entry.tier)
    return {
        **bench.dataset_config(dataset_id, artifact_root=artifact_root),
        "eval_episodes": int(protocol["eval_episodes"]),
        "eval_seed_base": int(protocol["eval_seed_base"]),
        "normalization": bench.environments[entry.env_id]["normalization"],
    }


def _identity(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": config["dataset_id"],
        "eval_episodes": config["eval_episodes"],
        "eval_seed_base": config["eval_seed_base"],
        "behavior_policy": config["behavior_policy"],
        "behavior_epsilon": float(config["behavior_epsilon"]),
        "normalization": config["normalization"],
    }


def is_current(bench: Benchmark, dataset_id: str, *, artifact_root: str | Path) -> bool:
    """True when a stored baseline exists and was computed with the current settings."""

    path = baseline_path(dataset_id, artifact_root)
    if not path.exists():
        return False
    config = baseline_config(bench, dataset_id, artifact_root=artifact_root)
    return load_json(path)["identity"] == _identity(config)


def evaluate_baselines(
    bench: Benchmark, dataset_id: str, *, artifact_root: str | Path, show_progress: bool = True
) -> dict[str, Any]:
    """Evaluate the low (random) and high (behavior) references and store them."""

    config = baseline_config(bench, dataset_id, artifact_root=artifact_root)
    seed = int(config["seed"])
    env = make_env(config, seed=seed, purpose="evaluate")
    try:
        policies = {
            "low": env.make_policy(config["normalization"]["low"], seed=seed, options=config),
            "high": behavior_policy(config, env)[0],
        }
    finally:
        env.close()
    references = {}
    for role, policy in policies.items():
        name = config["normalization"][role]
        # Sampled, not greedy: the references are the stochastic policies
        # (uniform random, epsilon-mixed behavior) that the scores compare to.
        summary = evaluate_policy(
            policy,
            config={**config, "method_id": f"baseline-{name}"},
            num_episodes=int(config["eval_episodes"]),
            eval_seed_base=int(config["eval_seed_base"]),
            label=f"baseline {name}",
            export_csv=False,
            show_progress=show_progress,
            deterministic=False,
        )["summary"]
        references[role] = {"policy": name, **summary}
    payload = {"schema_version": SCHEMA_VERSION, "identity": _identity(config), "references": references}
    save_json(payload, baseline_path(dataset_id, artifact_root))
    return payload


def load_references(dataset_id: str, artifact_root: str | Path) -> tuple[float, float]:
    """``(R_low, R_high)`` mean returns of a dataset's stored baselines."""

    path = baseline_path(dataset_id, artifact_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing baselines for {dataset_id!r}: run `projeto-rl baselines {dataset_id}` ({path})"
        )
    references = load_json(path)["references"]
    return float(references["low"]["mean_return"]), float(references["high"]["mean_return"])
