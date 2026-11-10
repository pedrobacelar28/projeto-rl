"""Rich-backed terminal output with JSON-safe machine output."""

from __future__ import annotations

import json
import sys
from typing import Any, Iterable

try:  # Keep diagnostics usable before dependencies are installed.
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:  # pragma: no cover - requirements install Rich
    Console = None
    Panel = None
    Table = None


class CliConsole:
    def __init__(self, *, json_mode: bool = False, color: str = "auto"):
        self.json_mode = json_mode
        self.color = color
        color_system = None if color == "never" else "auto"
        self._console = None if Console is None else Console(color_system=color_system, highlight=False)
        self._error_console = (
            None if Console is None else Console(stderr=True, color_system=color_system, highlight=False)
        )

    def emit(self, payload: Any) -> None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))

    def title(self, title: str, subtitle: str | None = None) -> None:
        if self.json_mode:
            return
        if self._console is None or Panel is None:
            print(title)
            if subtitle:
                print(subtitle)
            return
        self._console.print(Panel.fit(f"[bold cyan]{title}[/bold cyan]" + (f"\n{subtitle}" if subtitle else "")))

    def table(self, title: str, columns: Iterable[str], rows: Iterable[Iterable[Any]]) -> None:
        if self.json_mode:
            return
        rows = list(rows)
        if self._console is None or Table is None:
            print(title)
            print(" | ".join(columns))
            for row in rows:
                print(" | ".join(str(value) for value in row))
            return
        table = Table(title=title, header_style="bold cyan", expand=False)
        for column in columns:
            table.add_column(str(column), overflow="fold")
        for row in rows:
            table.add_row(*(str(value) for value in row))
        self._console.print(table)

    def note(self, message: str) -> None:
        if not self.json_mode:
            if self._console is None:
                print(message)
            else:
                self._console.print(message)

    def success(self, message: str) -> None:
        self.note(f"[green]OK[/green] {message}" if self._console is not None else f"OK {message}")

    def warning(self, message: str) -> None:
        self.note(f"[yellow]WARNING[/yellow] {message}" if self._console is not None else f"WARNING {message}")

    def error(self, message: str) -> None:
        if self.json_mode:
            self.emit({"error": message})
        elif self._console is None:
            print(f"error: {message}", file=sys.stderr)
        else:
            self._error_console.print(f"[bold red]error:[/bold red] {message}")
