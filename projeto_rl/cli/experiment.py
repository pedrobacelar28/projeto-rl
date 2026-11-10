"""`run-all`: one experiment end to end (datasets, baselines, training, summary), resumable."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from projeto_rl.cli import common
from projeto_rl.cli.research import experiment_jobs, print_summary
from projeto_rl.config import load_experiment
from projeto_rl.data.collect import collect_dataset
from projeto_rl.operations.artifacts import is_complete
from projeto_rl.operations.baselines import evaluate_baselines, is_current
from projeto_rl.operations.queue import execute_queue
from projeto_rl.operations.summary import summarize_runs
from projeto_rl.utils.progress import format_seconds

STAGES = ("datasets", "baselines", "training", "summary")


class _Clock:
    """Prints one progress line per event with stage and total elapsed time."""

    def __init__(self):
        self.started = time.monotonic()
        self.stage_started = self.started
        self.stage_seconds: dict[str, float] = {}
        self.stage = ""

    def begin(self, stage: str) -> None:
        self.stage, self.stage_started = stage, time.monotonic()
        self.line("started")

    def end(self) -> None:
        self.stage_seconds[self.stage] = time.monotonic() - self.stage_started
        self.line("done")

    def line(self, message: str) -> None:
        now = time.monotonic()
        print(
            f"[{STAGES.index(self.stage) + 1}/{len(STAGES)} {self.stage}] {message} | "
            f"stage {format_seconds(now - self.stage_started)} | total {format_seconds(now - self.started)}",
            file=sys.stderr,
            flush=True,
        )


def run_all(args) -> int:
    console = common.console(args)
    bench = common.benchmark(args)
    root = common.root(args)
    experiment = load_experiment(args.experiment)
    datasets = experiment["datasets"]
    jobs = experiment_jobs(args, experiment)
    missing_data = [d for d in datasets if not common.dataset_status(d, root)["ready"]]
    missing_baselines = [d for d in datasets if not is_current(bench, d, artifact_root=root)]
    pending_runs = [job for job in jobs if not is_complete(job.output_dir)]
    plan = {
        "experiment": str(experiment["path"]),
        "datasets_to_build": missing_data,
        "baselines_to_evaluate": missing_baselines,
        "runs_total": len(jobs),
        "runs_pending": len(pending_runs),
    }
    if args.dry_run:
        if args.json:
            console.emit(plan)
        else:
            console.title("run-all plan", experiment["description"])
            console.table("Work left", ("item", "value"), [(key, value) for key, value in plan.items()])
        return 0

    clock = _Clock()
    show_progress = sys.stderr.isatty()

    clock.begin("datasets")
    for index, dataset_id in enumerate(datasets, start=1):
        if dataset_id in missing_data:
            # A dataset of an older schema is rebuilt in place.
            clock.line(f"{index}/{len(datasets)} building {dataset_id}")
            config = bench.dataset_config(dataset_id, artifact_root=root, force=True)
            collect_dataset({**config, "show_progress": show_progress, "quiet": True})
        else:
            clock.line(f"{index}/{len(datasets)} {dataset_id} already built")
    clock.end()

    clock.begin("baselines")
    for index, dataset_id in enumerate(datasets, start=1):
        if dataset_id in missing_baselines:
            clock.line(f"{index}/{len(datasets)} evaluating random and behavior policies of {dataset_id}")
            evaluate_baselines(bench, dataset_id, artifact_root=root, show_progress=show_progress)
        else:
            clock.line(f"{index}/{len(datasets)} {dataset_id} baselines up to date")
    clock.end()

    clock.begin("training")
    _, manifest = execute_queue(
        jobs,
        root=root,
        max_workers=args.jobs,
        gpus=args.gpus,
        resume=True,
        follow=True,
        progress_interval_seconds=experiment["progress_interval_seconds"],
        run_started=clock.started,
    )
    clock.end()
    if manifest["failures"]:
        failed = [record["name"] for record in manifest["jobs"] if record["status"] == "failed"]
        raise RuntimeError(
            f"{len(failed)} runs failed (logs under {root / 'queues'}); run the same command again to retry them: "
            + ", ".join(failed)
        )

    clock.begin("summary")
    payload = summarize_runs(
        root, run_dirs=[Path(job.output_dir) for job in jobs], experiment=experiment, config_root=args.config_root
    )
    clock.end()
    if args.json:
        console.emit({key: payload[key] for key in ("output_dir", "aggregate", "comparison")})
    else:
        print_summary(console, payload)
        console.table(
            "Time per stage",
            ("stage", "time"),
            [
                *((stage, format_seconds(seconds)) for stage, seconds in clock.stage_seconds.items()),
                ("total", format_seconds(time.monotonic() - clock.started)),
            ],
        )
        console.note(f"Tables: {payload['output_dir']}/summary.md; loss curves: {payload['output_dir']}/loss_curves/")
    return 0


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "run-all", help="Run an experiment end to end: datasets, baselines, training, summary (resumable)."
    )
    parser.add_argument("experiment", help="configs/experiments/<name>.json")
    parser.add_argument("--jobs", type=int, default=1, help="Runs in parallel (they may share one GPU).")
    parser.add_argument("--gpus", help="Comma-separated CUDA indices, e.g. 0 or 0,1.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Show what is left to do and exit.")
    parser.set_defaults(handler=run_all, overrides=[], no_progress=True, force=False)
