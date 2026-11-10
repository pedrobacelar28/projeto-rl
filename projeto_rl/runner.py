"""Run one resolved experiment: offline training, then simulator evaluation.

Artifacts written to the run directory:

``config.json``            the fully resolved configuration
``run_manifest.json``      identity, provenance, and lifecycle status
``train_history.csv``      one row per validation check
``checkpoint_best.pt``     validation-selected checkpoint (deployed)
``checkpoint_last.pt``     final checkpoint
``eval_summary.json``      direct simulator evaluation of the best checkpoint
``episodes_summary.csv``   one row per evaluation episode
``trajectories.csv``       one row per evaluation step
``environment_spec.json``  the environment contract used for evaluation
``metrics_summary.json``   everything above that is scalar, in one document
"""

from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from projeto_rl.algorithms import get_algorithm, load_policy
from projeto_rl.evaluation import evaluate_policy
from projeto_rl.operations.artifacts import is_complete, write_manifest
from projeto_rl.training import train_offline
from projeto_rl.utils.io import ensure_dir, load_json, save_json
from projeto_rl.utils.progress import progress_write
from projeto_rl.utils.queue_progress import update_progress


@contextmanager
def _terminate_as_interrupt():
    """Turn SIGTERM into ``SystemExit`` so killed runs are recorded, not left 'running'."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_experiment(
    params: dict[str, Any],
    *,
    output_dir: str | Path,
    manifest: dict[str, Any] | None = None,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    output_dir = ensure_dir(output_dir)
    if not force and is_complete(output_dir):
        if not quiet:
            print(f"Already completed: {output_dir}")
        return output_dir
    show_progress = bool(params.get("show_progress", True))
    method = str(params.get("method_id", get_algorithm(params["algo"]).method_id))

    def phase(message: str) -> None:
        if show_progress:
            progress_write(f"[phase] {message}")

    save_json(params, output_dir / "config.json")
    if manifest is not None:
        manifest = write_manifest(output_dir, manifest, status="running")
    update_progress(
        status="running", phase="starting", method_id=method, total_steps=params.get("offline_train_steps"), step=0
    )
    try:
        with _terminate_as_interrupt():
            phase(f"optimization | algorithm={params['algo']}")
            result = train_offline(params)
            summary = result.summary
            phase(
                f"optimization complete | reason={summary['early_stop_reason']} best_step={summary['best_validation_step']} best_loss={summary['best_validation_loss']:.6f}"
            )
            pd.DataFrame(result.history).to_csv(output_dir / "train_history.csv", index=False)
            torch.save(result.best_checkpoint, output_dir / "checkpoint_best.pt")
            torch.save(result.last_checkpoint, output_dir / "checkpoint_last.pt")
            timing = {"optimization_seconds": summary["optimization_seconds"]}

            policy = load_policy(params, result.best_checkpoint)
            episodes = int(params["eval_episodes"])
            phase(f"direct evaluation | checkpoint=best episodes={episodes}")
            started = time.perf_counter()
            evaluation = evaluate_policy(
                policy,
                config=params,
                num_episodes=episodes,
                eval_seed_base=int(params["eval_seed_base"]),
                output_dir=output_dir,
                export_csv=bool(params.get("save_trajectories", True)),
                show_progress=show_progress,
            )
            eval_summary = evaluation["summary"]
            timing["evaluation_seconds"] = float(time.perf_counter() - started)
            timing["mean_action_latency_ms"] = eval_summary["mean_action_latency_ms"]
            save_json(eval_summary, output_dir / "eval_summary.json")
            phase(
                f"direct evaluation complete | return={eval_summary['mean_return']:.6f} success={eval_summary['mean_success']:.6f}"
            )

            save_json(
                {
                    "schema_version": 2,
                    "dataset_id": params.get("dataset_id"),
                    "env_id": params.get("env_id"),
                    "method_id": method,
                    "algo": params["algo"],
                    "representation": params["representation"],
                    "seed": int(params["seed"]),
                    "train_summary": summary,
                    "eval_summary": eval_summary,
                    "timing_summary": timing,
                },
                output_dir / "metrics_summary.json",
            )
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        if manifest is not None:
            write_manifest(output_dir, manifest, status=status, error=f"{type(exc).__name__}: {exc}")
        update_progress(status="failed", phase=status, method_id=method, error=f"{type(exc).__name__}: {exc}")
        raise
    if manifest is not None:
        write_manifest(output_dir, manifest, status="completed")
    update_progress(
        status="completed",
        phase="complete",
        method_id=method,
        step=summary["total_gradient_steps"],
        total_steps=params.get("offline_train_steps"),
        mean_return=eval_summary["mean_return"],
        mean_success=eval_summary["mean_success"],
    )
    if not quiet:
        print(f"Finished run: {output_dir}")
    return output_dir


def load_run(run_dir: str | Path) -> tuple[dict[str, Any], Any]:
    """Return ``(config, deployed_policy)`` for a completed run directory."""

    run_dir = Path(run_dir)
    checkpoint_path = run_dir / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")
    config = load_json(run_dir / "config.json")
    return config, load_policy(config, torch.load(checkpoint_path, map_location="cpu"))
