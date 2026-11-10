"""Repository locations shared by the CLI, configs, and artifact layout."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
