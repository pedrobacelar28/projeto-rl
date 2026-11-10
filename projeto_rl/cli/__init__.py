"""projeto-rl command-line interface.

Whole experiment::

    projeto-rl run-all configs/experiments/final.json --gpus 0 --jobs 8

Step by step::

    projeto-rl dataset build <dataset> -> projeto-rl baselines <dataset>
    -> projeto-rl train <dataset> <method> [--set representation='"standardize"']
    -> projeto-rl research summarize

Every command resolves its configuration from ``configs/`` and writes only
under the artifact root (``artifacts/`` by default).
"""

from __future__ import annotations

import argparse
import sys

from projeto_rl.cli import common, data, experiment, info, research, runs
from projeto_rl.config import ConfigError
from projeto_rl.paths import CONFIG_ROOT, DEFAULT_ARTIFACT_ROOT

GLOBAL_VALUE_FLAGS = {"--artifact-root", "--config-root", "--color"}


def _hoist_global_flags(argv: list[str]) -> list[str]:
    """Allow global flags before or after the subcommand."""

    leading, remaining, index = [], [], 0
    while index < len(argv):
        token = argv[index]
        if token == "--json":
            leading.append(token)
        elif token in GLOBAL_VALUE_FLAGS and index + 1 < len(argv):
            leading += [token, argv[index + 1]]
            index += 1
        else:
            remaining.append(token)
        index += 1
    return [*leading, *remaining]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="projeto-rl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--config-root", default=str(CONFIG_ROOT))
    parser.add_argument("--color", choices=["auto", "never"], default="auto")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for module in (info, data, runs, research, experiment):
        module.register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(_hoist_global_flags(list(sys.argv[1:] if argv is None else argv)))
    try:
        return int(args.handler(args))
    except (ConfigError, FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        common.console(args).error(str(exc).strip("'\""))
        return 2
