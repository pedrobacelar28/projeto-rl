"""Aggregate an experiment's runs: normalized scores and the paired comparison against a reference representation.

The normalized score of a run is ``100 * (R - R_low) / (R_high - R_low)``,
with ``R_low`` the random policy and ``R_high`` the dataset's behavior policy
(see :mod:`projeto_rl.operations.baselines`). Every representation is compared to
the experiment's ``reference_representation`` on the same dataset, method, and
seeds, so each comparison is a paired difference over seeds. Raw returns of
different datasets are never averaged. Groups of runs are aggregated with the
interquartile mean (IQM) of the normalized score, which a single diverged run
cannot dominate (Agarwal et al., "Deep Reinforcement Learning at the Edge of
the Statistical Precipice", NeurIPS 2021).
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy import stats

from projeto_rl.config import Benchmark
from projeto_rl.operations.artifacts import artifact_root
from projeto_rl.operations.baselines import load_references
from projeto_rl.operations.plots import plot_loss_curves
from projeto_rl.utils.io import ensure_dir, load_json, save_json

TIMING_FIELDS = ("optimization_seconds", "evaluation_seconds")


def normalized_score(value: float, low: float, high: float) -> float:
    if high == low:
        raise ValueError(f"Cannot normalize: the two reference returns are equal ({low})")
    return 100.0 * (float(value) - float(low)) / (float(high) - float(low))


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot aggregate an empty group")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def paired_difference(treatment: dict[int, float], control: dict[int, float]) -> dict[str, Any]:
    """Mean and 95% Student-t interval of ``treatment[seed] - control[seed]`` over shared seeds.

    Both sides must have exactly the same seeds; with one seed the interval is
    undefined (NaN).
    """

    if set(treatment) != set(control):
        raise ValueError(f"Unpaired seeds: {sorted(treatment)} vs {sorted(control)}")
    seeds = sorted(treatment)
    differences = [treatment[seed] - control[seed] for seed in seeds]
    mean, std = _mean_std(differences)
    if len(seeds) > 1:
        half_width = float(stats.t.ppf(0.975, len(seeds) - 1)) * std / math.sqrt(len(seeds))
        low, high = mean - half_width, mean + half_width
    else:
        low = high = float("nan")
    return {
        "num_seeds": len(seeds),
        "mean_difference": mean,
        "ci95_low": low,
        "ci95_high": high,
        "wins": sum(difference > 0 for difference in differences),
    }


def run_row(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    """One flat row for a completed run, or ``None`` for runs that did not complete."""

    if manifest.get("status") != "completed":
        return None
    metrics = load_json(run_dir / "metrics_summary.json")
    evaluation = metrics["eval_summary"]
    timing = metrics["timing_summary"]
    return {
        "run_dir": str(run_dir),
        "run_id": manifest["run_id"],
        "dataset_id": manifest["dataset_id"],
        "env_id": manifest["env_id"],
        "method_id": manifest["method_id"],
        "representation": manifest["representation"],
        "seed": int(manifest["seed"]),
        "mean_return": float(evaluation["mean_return"]),
        "mean_success": float(evaluation["mean_success"]),
        **{key: float(timing[key]) for key in TIMING_FIELDS},
    }


def interquartile_mean(values: list[float]) -> float:
    """Mean of the middle 50% of ``values`` (25% trimmed from each end, as ``scipy.stats.trim_mean``)."""

    if not values:
        raise ValueError("Cannot aggregate an empty group")
    return float(stats.trim_mean(values, 0.25))


def summarize(
    rows: list[dict[str, Any]],
    *,
    references: dict[str, tuple[float, float]],
    groups: dict[str, str],
    reference_representation: str,
):
    """Scores, per-method statistics, paired comparisons, and aggregates.

    ``references`` maps each dataset to ``(R_low, R_high)``; ``groups`` maps
    each environment to its group (``homogeneous`` or ``heterogeneous``).
    """

    scored = [
        {**row, "normalized_score": normalized_score(row["mean_return"], *references[row["dataset_id"]])}
        for row in rows
    ]
    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in scored:
        cells[(row["dataset_id"], row["method_id"], row["representation"])].append(row)

    by_method = []
    for (dataset_id, method_id, representation), group in sorted(cells.items()):
        item: dict[str, Any] = {
            "dataset_id": dataset_id,
            "method_id": method_id,
            "representation": representation,
            "num_seeds": len(group),
        }
        for field in ("normalized_score", "mean_return", "mean_success", *TIMING_FIELDS):
            item[f"{field}_mean"], item[f"{field}_std"] = _mean_std([row[field] for row in group])
        by_method.append(item)

    comparison = []
    for (dataset_id, method_id, representation), group in sorted(cells.items()):
        control = cells.get((dataset_id, method_id, reference_representation))
        if representation == reference_representation or control is None:
            continue
        comparison.append(
            {
                "dataset_id": dataset_id,
                "method_id": method_id,
                "representation": representation,
                **paired_difference(
                    {row["seed"]: row["normalized_score"] for row in group},
                    {row["seed"]: row["normalized_score"] for row in control},
                ),
            }
        )

    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in scored:
        for group in (row["dataset_id"], groups[row["env_id"]], "all"):
            buckets[(group, row["representation"])].append(row["normalized_score"])
    order = {name: index for index, name in enumerate(sorted({row["dataset_id"] for row in scored}))}
    order.update({"homogeneous": len(order), "heterogeneous": len(order) + 1, "all": len(order) + 2})
    aggregate = [
        {
            "group": group,
            "representation": representation,
            "iqm_normalized_score": interquartile_mean(values),
            "num_runs": len(values),
        }
        for (group, representation), values in sorted(buckets.items(), key=lambda item: (order[item[0][0]], item[0][1]))
    ]
    return {
        "reference_representation": reference_representation,
        "rows": scored,
        "by_method": by_method,
        "comparison": comparison,
        "aggregate": aggregate,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(summary: dict[str, Any], plots: list[Path], destination: Path) -> str:
    lines = [
        "# Representation comparison",
        "",
        "Normalized score: 0 = random policy, 100 = the behavior policy that logged the dataset.",
        "",
        "## Normalized score per group (IQM over runs)",
        "",
        "| Group | Representation | IQM score | Runs |",
        "|---|---|---:|---:|",
    ]
    lines += [
        f"| {item['group']} | `{item['representation']}` | {item['iqm_normalized_score']:.1f} | {item['num_runs']} |"
        for item in summary["aggregate"]
    ]
    lines += [
        "",
        f"## Paired difference against `{summary['reference_representation']}` (normalized score, same seeds)",
        "",
        "| Dataset | Method | Representation | Mean difference | 95% CI | Wins |",
        "|---|---|---|---:|---:|---:|",
    ]
    lines += [
        f"| `{item['dataset_id']}` | `{item['method_id']}` | `{item['representation']}` | "
        f"{item['mean_difference']:+.1f} | [{item['ci95_low']:.1f}, {item['ci95_high']:.1f}] | "
        f"{item['wins']}/{item['num_seeds']} |"
        for item in summary["comparison"]
    ]
    lines += [
        "",
        "## Per method",
        "",
        "| Dataset | Method | Representation | Seeds | Score (mean ± std) | Return (mean ± std) | Train time (s) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| `{item['dataset_id']}` | `{item['method_id']}` | `{item['representation']}` | {item['num_seeds']} | "
        f"{item['normalized_score_mean']:.1f} ± {item['normalized_score_std']:.1f} | "
        f"{item['mean_return_mean']:.3f} ± {item['mean_return_std']:.3f} | {item['optimization_seconds_mean']:.0f} |"
        for item in summary["by_method"]
    ]
    lines += ["", "## Loss curves", ""]
    lines += [f"![{path.stem}]({path.relative_to(destination)})" for path in plots]
    return "\n".join(lines) + "\n"


def summarize_runs(
    root: str | Path | None,
    *,
    run_dirs: list[Path],
    experiment: dict[str, Any],
    output_dir: str | Path | None = None,
    config_root: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize one experiment's runs (``run_dirs``) into CSV/JSON/Markdown reports and loss plots.

    Taking the experiment's run directories, instead of every run under
    ``root``, keeps runs of older protocols out of the tables.
    """

    bench = Benchmark.load(config_root)
    rows = [row for run_dir in run_dirs if (row := run_row(run_dir, load_json(run_dir / "run_manifest.json")))]
    if len(rows) != len(run_dirs):
        raise ValueError(f"Only {len(rows)} of the experiment's {len(run_dirs)} runs are completed")
    rows.sort(key=lambda row: (row["dataset_id"], row["method_id"], row["representation"], row["seed"]))
    references = {
        dataset_id: load_references(dataset_id, artifact_root(root))
        for dataset_id in {row["dataset_id"] for row in rows}
    }
    groups = {env_id: str(spec["feature_mix"]) for env_id, spec in bench.environments.items()}
    summary = summarize(
        rows, references=references, groups=groups, reference_representation=experiment["reference_representation"]
    )
    destination = ensure_dir(
        output_dir
        or artifact_root(root) / "reports" / "summary" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    for name in ("rows", "by_method", "comparison"):
        _write_csv(destination / f"{'runs' if name == 'rows' else name}.csv", summary[name])
    plots = plot_loss_curves(rows, destination / "loss_curves", representations=experiment["representations"])
    payload = {
        "schema_version": 4,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(destination),
        "references": {dataset_id: {"low": low, "high": high} for dataset_id, (low, high) in references.items()},
        "loss_plots": [str(path) for path in plots],
        **summary,
    }
    save_json(payload, destination / "summary.json")
    (destination / "summary.md").write_text(_markdown(summary, plots, destination), encoding="utf-8")
    return payload
