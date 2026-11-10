"""`dataset list|show|build`, `inspect dataset`, and `baselines`."""

from __future__ import annotations

import numpy as np

from projeto_rl.cli import common
from projeto_rl.data.collect import collect_dataset
from projeto_rl.operations.baselines import baseline_path, evaluate_baselines


def _dataset_rows(datasets):
    return [
        (
            item["id"],
            item["behavior_policy"],
            common.compact_count(item["train_episodes"]),
            "ready" if item["availability"]["ready"] else "build required",
        )
        for item in datasets
    ]


def dataset_list(args) -> int:
    bench = common.benchmark(args)
    console = common.console(args)
    datasets = [common.dataset_payload(bench, common.root(args), entry.identifier) for entry in bench.datasets]
    if args.json:
        console.emit({"datasets": datasets})
    else:
        console.title("Offline Datasets", "Use `dataset show <id>` for splits and compatible methods.")
        console.table("Datasets", ("id", "behavior policy", "train episodes", "state"), _dataset_rows(datasets))
    return 0


def dataset_show(args) -> int:
    bench = common.benchmark(args)
    console = common.console(args)
    payload = common.dataset_payload(bench, common.root(args), args.dataset_id)
    if args.json:
        console.emit(payload)
        return 0
    console.title(f"Dataset: {payload['id']}", payload["description"])
    console.table(
        "Identity",
        ("field", "value"),
        [
            (key, value)
            for key, value in payload.items()
            if key not in {"availability", "environment_spec", "splits", "methods", "description"}
        ],
    )
    console.table("Files", ("split", "path"), payload["availability"]["paths"].items())
    console.table("Methods", ("method",), [(name,) for name in payload["methods"]])
    return 0


def dataset_build(args) -> int:
    bench = common.benchmark(args)
    console = common.console(args)
    config = bench.dataset_config(args.dataset_id, artifact_root=common.root(args), force=args.force)
    if args.dry_run:
        if args.json:
            console.emit({"dataset_id": args.dataset_id, "config": config})
        else:
            console.title("Dataset Build Plan", args.dataset_id)
            console.table("Configuration", ("field", "value"), config.items())
            console.note("Dry run only. No data will be generated.")
        return 0
    metadata = collect_dataset({**config, "show_progress": not args.no_progress, "quiet": bool(args.json)})
    if args.json:
        console.emit({"dataset_id": args.dataset_id, "config": config, "metadata": metadata})
    else:
        console.success(f"Dataset ready at {config['output_dir']}")
    return 0


def inspect_dataset(args) -> int:
    bench = common.benchmark(args)
    console = common.console(args)
    payload = common.dataset_payload(bench, common.root(args), args.dataset_id)
    status = payload["availability"]
    if status["ready"]:
        train = np.load(status["paths"]["train"])
        payload["arrays"] = {
            key: {"shape": list(train[key].shape), "dtype": str(train[key].dtype)} for key in sorted(train.files)
        }
        payload["transitions"] = int(train["actions"].shape[0])
        payload["episodes"] = int(np.unique(train["episode_ids"]).size)
    if args.json:
        console.emit(payload)
    elif not status["ready"]:
        console.warning(f"Dataset is missing: {status['missing']}")
    else:
        console.title("Dataset Inspection", args.dataset_id)
        console.table(
            "Arrays",
            ("array", "shape", "dtype"),
            [(key, value["shape"], value["dtype"]) for key, value in payload["arrays"].items()],
        )
    return 0


def baselines(args) -> int:
    bench = common.benchmark(args)
    console = common.console(args)
    payload = evaluate_baselines(
        bench, args.dataset_id, artifact_root=common.root(args), show_progress=not args.no_progress
    )
    if args.json:
        console.emit(payload)
    else:
        console.success(f"Baselines written to {baseline_path(args.dataset_id, common.root(args))}")
        console.table(
            "Reference policies",
            ("role", "policy", "return", "success", "length"),
            [
                (role, ref["policy"], ref["mean_return"], ref["mean_success"], ref["mean_length"])
                for role, ref in payload["references"].items()
            ],
        )
    return 0


def register(subparsers) -> None:
    dataset = subparsers.add_parser("dataset", help="List, inspect, or build offline datasets.").add_subparsers(
        dest="dataset_command", required=True
    )
    dataset.add_parser("list", help="List datasets.").set_defaults(handler=dataset_list)
    show = dataset.add_parser("show", help="Show one dataset and its availability.")
    show.add_argument("dataset_id")
    show.set_defaults(handler=dataset_show)
    build = dataset.add_parser("build", help="Collect the train and validation splits.")
    build.add_argument("dataset_id")
    build.add_argument("--force", action="store_true")
    build.add_argument("--dry-run", action="store_true")
    build.add_argument("--no-progress", action="store_true")
    build.set_defaults(handler=dataset_build)

    parser = subparsers.add_parser(
        "baselines", help="Evaluate the random and behavior policies that normalize a dataset's scores."
    )
    parser.add_argument("dataset_id")
    parser.add_argument("--no-progress", action="store_true")
    parser.set_defaults(handler=baselines)
