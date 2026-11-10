"""`research sweep|matrix|summarize|reevaluate`."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from projeto_rl.cli import common
from projeto_rl.cli.runs import evaluate, queue_jobs, run_queue
from projeto_rl.config import load_experiment
from projeto_rl.operations.summary import summarize_runs


def sweep(args) -> int:
    jobs = queue_jobs(
        args, dataset_ids=[args.dataset_id], methods=[args.method_id], seeds=common.parse_seeds(args.seeds)
    )
    return run_queue(args, jobs, title=f"Sweep: {args.dataset_id} {args.method_id}")


def experiment_jobs(args, experiment: dict) -> list:
    """One queue job per dataset x method x representation x seed of ``experiment``."""

    return queue_jobs(
        args,
        dataset_ids=experiment["datasets"],
        methods=experiment["methods"],
        seeds=experiment["seeds"],
        representations=experiment["representations"],
    )


def matrix(args) -> int:
    experiment = load_experiment(args.definition)
    return run_queue(args, experiment_jobs(args, experiment), title=f"Matrix: {experiment['path'].name}")


def summarize(args) -> int:
    console = common.console(args)
    experiment = load_experiment(args.experiment)
    payload = summarize_runs(
        common.root(args),
        run_dirs=[Path(job.output_dir) for job in experiment_jobs(args, experiment)],
        experiment=experiment,
        output_dir=args.output_dir,
        config_root=args.config_root,
    )
    if args.json:
        console.emit({key: payload[key] for key in ("output_dir", "aggregate", "comparison")})
    else:
        print_summary(console, payload)
    return 0


def _interval(item: dict) -> str:
    # Rich reads "[...]" as markup, so the interval is written as "low to high".
    if math.isnan(item["ci95_low"]):
        return "n/a (1 seed)"
    return f"{item['ci95_low']:.1f} to {item['ci95_high']:.1f}"


def print_summary(console, payload: dict) -> None:
    console.success(f"Summary of {len(payload['rows'])} runs written to {payload['output_dir']}")
    console.table(
        "Normalized score, IQM over runs (0 = random, 100 = behavior policy)",
        ("group", "representation", "IQM score", "runs"),
        [
            (item["group"], item["representation"], f"{item['iqm_normalized_score']:.1f}", item["num_runs"])
            for item in payload["aggregate"]
        ],
    )
    console.table(
        "Per method (mean ± std over seeds)",
        ("dataset", "method", "representation", "seeds", "score", "return"),
        [
            (
                item["dataset_id"],
                item["method_id"],
                item["representation"],
                item["num_seeds"],
                f"{item['normalized_score_mean']:.1f} ± {item['normalized_score_std']:.1f}",
                f"{item['mean_return_mean']:.3f} ± {item['mean_return_std']:.3f}",
            )
            for item in payload["by_method"]
        ],
    )
    if payload["comparison"]:
        console.table(
            f"Paired difference against {payload['reference_representation']} (normalized score, same seeds)",
            ("dataset", "method", "representation", "mean diff", "CI 95%", "wins"),
            [
                (
                    item["dataset_id"],
                    item["method_id"],
                    item["representation"],
                    f"{item['mean_difference']:+.1f}",
                    _interval(item),
                    f"{item['wins']}/{item['num_seeds']}",
                )
                for item in payload["comparison"]
            ],
        )


def reevaluate(args) -> int:
    for reference in args.runs:
        evaluate(argparse.Namespace(**{**vars(args), "run": reference}))
    return 0


def register(subparsers) -> None:
    research = subparsers.add_parser("research", help="Sweeps, experiment matrices, and summaries.").add_subparsers(
        dest="research_command", required=True
    )

    parser = research.add_parser("sweep", help="Queue one method on one dataset over seeds.")
    parser.add_argument("dataset_id")
    parser.add_argument("method_id")
    common.add_queue_options(parser, default_seeds="0,1,2")
    common.add_run_options(parser)
    parser.set_defaults(handler=sweep)

    parser = research.add_parser(
        "matrix", help="Queue an experiment: datasets x methods x representations x seeds (configs/experiments/*.json)."
    )
    parser.add_argument("definition")
    common.add_queue_options(parser, default_seeds=None)
    common.add_run_options(parser)
    parser.set_defaults(handler=matrix)

    parser = research.add_parser(
        "summarize",
        help="Normalized scores, paired comparison against the reference representation, and loss plots of an "
        "experiment's runs.",
    )
    parser.add_argument("experiment", help="configs/experiments/<name>.json")
    parser.add_argument("--output-dir", help="Defaults to <artifact-root>/reports/summary.")
    parser.set_defaults(handler=summarize, overrides=[], no_progress=True, force=False, device="auto")

    parser = research.add_parser("reevaluate", help="Fresh simulator evaluation for several runs.")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.set_defaults(handler=reevaluate)
