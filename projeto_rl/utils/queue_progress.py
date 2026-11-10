"""Atomic progress side-channel used by queued benchmark workers."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from projeto_rl.utils.io import ensure_dir, load_json, save_json


def _configured_path(path: str | Path | None = None) -> Path | None:
    value = path if path is not None else os.environ.get("PROJETO_RL_PROGRESS_PATH")
    if not value:
        return None
    return Path(value)


def update_progress(path: str | Path | None = None, **fields: Any) -> bool:
    """Write one progress snapshot, atomically, when a queue path is configured.

    Direct ``train`` calls do not set ``PROJETO_RL_PROGRESS_PATH`` and therefore keep
    their existing behavior. Queue workers set it for the duration of a job.
    """

    destination = _configured_path(path)
    if destination is None:
        return False
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    ensure_dir(destination.parent)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    save_json(payload, temporary)
    temporary.replace(destination)
    return True


def read_progress(path: str | Path | None) -> dict[str, Any]:
    """Read a progress snapshot, returning an empty object if unavailable."""

    if not path:
        return {}
    try:
        return load_json(Path(path))
    except (FileNotFoundError, OSError, ValueError):
        # The writer replaces files atomically, but a reader may still observe
        # a missing snapshot during process startup or cleanup.
        return {}
