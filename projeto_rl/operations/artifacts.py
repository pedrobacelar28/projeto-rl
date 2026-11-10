"""Artifact layout, run manifests, and run discovery.

Runs live at ``<artifact_root>/runs/<dataset_id>/<method_id>/<representation>/seed-<n>--<hash>``,
where ``hash`` fingerprints the fully resolved configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from projeto_rl.paths import DEFAULT_ARTIFACT_ROOT, PROJECT_ROOT
from projeto_rl.utils.io import ensure_dir, load_json, save_json

# Execution-only keys that do not change results.
SIGNATURE_IGNORED_KEYS = {"show_progress"}
FINAL_STATUSES = {"completed", "failed", "interrupted"}


def artifact_root(value: str | Path | None = None) -> Path:
    return Path(value or DEFAULT_ARTIFACT_ROOT).resolve()


def _normalize(value):
    if isinstance(value, float):
        return float(f"{value:.10g}")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def experiment_signature(params: dict[str, Any]) -> str:
    normalized = {key: _normalize(value) for key, value in sorted(params.items()) if key not in SIGNATURE_IGNORED_KEYS}
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def run_directory(params: dict[str, Any], *, root: str | Path | None = None) -> Path:
    signature = experiment_signature(params)[:12]
    return (
        artifact_root(root)
        / "runs"
        / Path(params["dataset_id"])
        / params["method_id"]
        / params["representation"]
        / f"seed-{int(params['seed'])}--{signature}"
    )


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _provenance(device: str | None) -> dict[str, Any]:
    cuda = bool(torch.cuda.is_available())
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda if cuda else None,
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if cuda
        else [],
        "requested_device": device or "auto",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def initial_manifest(params: dict[str, Any], *, run_dir: Path, device: str | None = None) -> dict[str, Any]:
    signature = experiment_signature(params)
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 3,
        "run_id": (
            f"{params['dataset_id']}:{params['method_id']}:{params['representation']}"
            f":seed-{int(params['seed'])}:{signature[:12]}"
        ),
        "status": "planned",
        "created_at": now,
        "updated_at": now,
        "artifact_dir": str(run_dir),
        "dataset_id": params["dataset_id"],
        "env_id": params["env_id"],
        "method_id": params["method_id"],
        "algo": params["algo"],
        "representation": params["representation"],
        "seed": int(params["seed"]),
        "config_signature": signature,
        "provenance": _provenance(device),
    }


def write_manifest(
    run_dir: str | Path, manifest: dict[str, Any], *, status: str | None = None, error: str | None = None
) -> dict[str, Any]:
    run_dir = ensure_dir(run_dir)
    current = dict(manifest)
    if status is not None:
        current["status"] = status
    if error is not None:
        current["error"] = error
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    if status == "running":
        # Lets `status` tell a live run from one whose process died.
        current["process"] = {"pid": os.getpid(), "host": socket.gethostname()}
    if status in FINAL_STATUSES:
        current["finished_at"] = current["updated_at"]
    save_json(current, run_dir / "run_manifest.json")
    return current


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def effective_status(manifest: dict[str, Any]) -> str:
    """Manifest status, reporting ``stale`` for a 'running' run whose process is gone."""

    status = str(manifest.get("status", "unknown"))
    process = manifest.get("process") or {}
    if (
        status == "running"
        and process.get("host") == socket.gethostname()
        and not _process_alive(int(process.get("pid", -1)))
    ):
        return "stale"
    return status


def load_manifest(run_dir: str | Path) -> dict[str, Any]:
    return load_json(Path(run_dir) / "run_manifest.json")


def is_complete(run_dir: str | Path) -> bool:
    path = Path(run_dir)
    if not (path / "run_manifest.json").exists() or not (path / "eval_summary.json").exists():
        return False
    try:
        return load_manifest(path).get("status") == "completed"
    except (OSError, ValueError):
        return False


def discover_runs(root: str | Path | None = None) -> list[tuple[Path, dict[str, Any]]]:
    runs_root = artifact_root(root) / "runs"
    if not runs_root.exists():
        return []
    discovered = []
    for path in runs_root.rglob("run_manifest.json"):
        try:
            discovered.append((path.parent, load_json(path)))
        except (OSError, ValueError):
            continue
    return sorted(discovered, key=lambda item: item[1].get("updated_at", ""), reverse=True)


def resolve_run(reference: str, root: str | Path | None = None) -> Path:
    candidate = Path(reference)
    if candidate.exists() and (candidate / "run_manifest.json").exists():
        return candidate.resolve()
    matches = [
        path for path, manifest in discover_runs(root) if manifest.get("run_id") == reference or path.name == reference
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No run matches {reference!r} under {artifact_root(root) / 'runs'}")
    raise ValueError(f"Run reference {reference!r} is ambiguous; use its full path or run_id")
