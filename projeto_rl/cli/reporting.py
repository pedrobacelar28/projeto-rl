"""Compact, environment-agnostic reports for one completed run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from projeto_rl.operations.artifacts import artifact_root, effective_status, load_manifest
from projeto_rl.utils.io import ensure_dir, load_json, save_json


def _read_optional(path: Path) -> dict[str, Any]:
    return load_json(path) if path.exists() else {}


def build_run_report(run_dir: str | Path, *, root: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    metrics = _read_optional(run_dir / "metrics_summary.json")
    evaluation = _read_optional(run_dir / "eval_summary.json")
    destination = ensure_dir(artifact_root(root) / "reports" / run_dir.name)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "run_id": manifest.get("run_id"),
            "status": effective_status(manifest),
            "dataset_id": manifest.get("dataset_id"),
            "method_id": manifest.get("method_id"),
            "representation": manifest.get("representation"),
            "environment": manifest.get("env_id"),
            "seed": manifest.get("seed"),
            "artifact_dir": str(run_dir),
        },
        "optimization": metrics.get("train_summary", {}),
        "timing": metrics.get("timing_summary", {}),
        "direct_evaluation": evaluation,
    }
    save_json(summary, destination / "summary.json")
    lines = [
        f"# Run Report: {summary['run']['run_id']}",
        "",
        "## Identity",
        f"- Status: `{summary['run']['status']}`",
        f"- Dataset: `{summary['run']['dataset_id']}`",
        f"- Method: `{summary['run']['method_id']}`",
        f"- Representation: `{summary['run']['representation']}`",
        f"- Environment: `{summary['run']['environment']}`",
        f"- Seed: `{summary['run']['seed']}`",
        "",
        "## Offline Optimization",
        "```json",
        json.dumps(summary["optimization"], indent=2, sort_keys=True),
        "```",
        "",
        "## Direct Simulator Evaluation",
        "```json",
        json.dumps(summary["direct_evaluation"], indent=2, sort_keys=True),
        "```",
        "",
        "## Cost Accounting",
        "```json",
        json.dumps(summary["timing"], indent=2, sort_keys=True),
        "```",
    ]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination, summary
