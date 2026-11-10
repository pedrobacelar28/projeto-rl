"""Drop-in progress bars backed by Rich in interactive terminals."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from tqdm.auto import tqdm as _tqdm

try:  # Rich remains an optional import for library-only callers.
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except ImportError:  # pragma: no cover - requirements install Rich
    Console = None
    Progress = None


_progress: Any | None = None
_active_tasks = 0


def _interactive(disable: bool) -> bool:
    return not disable and sys.stderr.isatty() and Progress is not None


def _acquire() -> Any:
    global _progress
    if _progress is None:
        _progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=Console(stderr=True) if Console is not None else None,
            transient=False,
        )
        _progress.start()
    return _progress


def _release() -> None:
    global _progress
    if _progress is not None and _active_tasks == 0:
        _progress.stop()
        _progress = None


@dataclass
class _RichBar:
    iterable: Iterable[Any] | None
    total: int | None
    desc: str
    unit: str
    disable: bool
    leave: bool = True

    def __post_init__(self) -> None:
        global _active_tasks
        self._fallback = None
        self._task_id = None
        self._postfix = ""
        if not _interactive(self.disable):
            self._fallback = _tqdm(
                self.iterable,
                total=self.total,
                desc=self.desc,
                unit=self.unit,
                dynamic_ncols=True,
                leave=self.leave,
                disable=True,
            )
            return
        progress = _acquire()
        self._task_id = progress.add_task(self.desc, total=self.total)
        _active_tasks += 1

    def __iter__(self) -> Iterator[Any]:
        if self.iterable is None:
            return iter(())
        if self._fallback is not None:
            yield from self._fallback
            return
        for value in self.iterable:
            yield value
            self.update(1)
        self.close()

    def update(self, amount: int = 1) -> None:
        if self._fallback is not None:
            self._fallback.update(amount)
        elif self._task_id is not None and _progress is not None:
            _progress.update(self._task_id, advance=amount)

    def set_postfix(self, ordered_dict: dict[str, Any] | None = None, **kwargs: Any) -> None:
        values = dict(ordered_dict or {})
        values.update(kwargs)
        self._postfix = "  " + " ".join(f"{key}={value}" for key, value in values.items()) if values else ""
        if self._fallback is not None:
            self._fallback.set_postfix(values)
        elif self._task_id is not None and _progress is not None:
            _progress.update(self._task_id, description=f"{self.desc}{self._postfix}")

    def close(self) -> None:
        global _active_tasks
        if self._fallback is not None:
            self._fallback.close()
            self._fallback = None
            return
        if self._task_id is not None and _progress is not None:
            if self.leave:
                if self.total is not None:
                    _progress.update(self._task_id, completed=self.total)
            else:
                _progress.remove_task(self._task_id)
            self._task_id = None
            _active_tasks = max(0, _active_tasks - 1)
            _release()


def tqdm(iterable: Iterable[Any] | None = None, *args: Any, **kwargs: Any) -> _RichBar:
    del args
    total = kwargs.pop("total", None)
    if total is None and iterable is not None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            pass
    return _RichBar(
        iterable=iterable,
        total=total,
        desc=str(kwargs.pop("desc", "progress")),
        unit=str(kwargs.pop("unit", "it")),
        disable=bool(kwargs.pop("disable", False)),
        leave=bool(kwargs.pop("leave", True)),
    )


def trange(*args: Any, **kwargs: Any) -> _RichBar:
    return tqdm(range(*args), **kwargs)


def progress_write(message: str) -> None:
    if _progress is not None:
        _progress.console.print(message)
    else:
        print(message)


def format_seconds(seconds: float) -> str:
    """``3725`` -> ``"1h02m05s"``; ``65`` -> ``"1m05s"``; ``9`` -> ``"9s"``."""

    total = int(round(max(0.0, float(seconds))))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
