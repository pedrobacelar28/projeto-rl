"""Loss curves of an experiment's runs: one PNG per dataset, train and validation loss per method."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # files only; never opens a window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from projeto_rl.utils.io import ensure_dir, load_json  # noqa: E402

# Categorical hues in fixed order; a representation keeps its color across
# plots because colors follow the experiment's representation list.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE = "#fcfcfb"
INK = "#3d3d3a"
MUTED_INK = "#73726c"
GRID = "#e5e4df"


def _history(run_dir: str | Path) -> tuple[pd.DataFrame, int]:
    run_dir = Path(run_dir)
    history = pd.read_csv(run_dir / "train_history.csv")
    best_step = int(load_json(run_dir / "metrics_summary.json")["train_summary"]["best_validation_step"])
    return history, best_step


def _style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED_INK)
    axis.tick_params(colors=MUTED_INK, labelsize=8)


def plot_loss_curves(rows: list[dict[str, Any]], output_dir: Path, *, representations: list[str]) -> list[Path]:
    """Write ``<output_dir>/<dataset>.png`` for every dataset in ``rows`` and return the paths.

    Each row of panels is one method; the left panel is the training loss and
    the right one the validation loss that selects the checkpoint (dot). Every
    line is one seed, colored by representation. The y axis is logarithmic when
    every loss in the row is positive, so a diverging run stays readable next
    to a converging one.
    """

    if len(representations) > len(SERIES_COLORS):
        raise ValueError(f"At most {len(SERIES_COLORS)} representations can be plotted, got {len(representations)}")
    colors = dict(zip(representations, SERIES_COLORS))
    by_dataset: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_dataset[row["dataset_id"]][row["method_id"]].append(row)

    ensure_dir(output_dir)
    paths = []
    for dataset_id, methods in sorted(by_dataset.items()):
        figure, axes = plt.subplots(
            len(methods), 2, figsize=(10, 2.6 * len(methods)), sharex=True, squeeze=False, facecolor=SURFACE
        )
        for row_axes, (method_id, runs) in zip(axes, sorted(methods.items())):
            histories = [(run, *_history(run["run_dir"])) for run in runs]
            positive = all(
                (history[column] > 0).all()
                for _, history, _ in histories
                for column in ("train_loss", "validation_loss")
            )
            for axis, column, title in (
                (row_axes[0], "train_loss", "train loss"),
                (row_axes[1], "validation_loss", "validation loss"),
            ):
                _style(axis)
                for run, history, best_step in histories:
                    color = colors[run["representation"]]
                    axis.plot(history["step"], history[column], color=color, linewidth=1.4, alpha=0.85)
                    if column == "validation_loss":
                        best = history.loc[history["step"] == best_step, column]
                        axis.plot(
                            [best_step],
                            best.to_numpy(),
                            marker="o",
                            markersize=6,
                            color=color,
                            markeredgecolor=SURFACE,
                            markeredgewidth=1.5,
                            linestyle="none",
                        )
                if positive:
                    axis.set_yscale("log")
                axis.set_title(f"{method_id}: {title}", fontsize=9, color=INK, loc="left")
        for axis in axes[-1]:
            axis.set_xlabel("gradient step", fontsize=8, color=MUTED_INK)
        handles = [Line2D([], [], color=colors[name], linewidth=2, label=name) for name in representations]
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                color=MUTED_INK,
                markeredgecolor=SURFACE,
                linestyle="none",
                label="selected checkpoint",
            )
        )
        figure.legend(handles=handles, loc="upper right", fontsize=8, frameon=False, ncol=len(handles))
        figure.suptitle(f"{dataset_id}: one line per seed", x=0.01, ha="left", fontsize=10, color=INK)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        path = output_dir / f"{dataset_id.replace('/', '__')}.png"
        figure.savefig(path, dpi=120, facecolor=SURFACE)
        plt.close(figure)
        paths.append(path)
    return paths
