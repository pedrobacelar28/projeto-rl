"""`train`, `suite`, `status`, `inspect`, `evaluate`, and `report`."""

from __future__ import annotations

from projeto_rl.cli import common
from projeto_rl.cli.data import inspect_dataset
from projeto_rl.cli.reporting import build_run_report
from projeto_rl.evaluation import evaluate_policy
from projeto_rl.operations.artifacts import discover_runs, effective_status, load_manifest, resolve_run
from projeto_rl.operations.queue import QueueJob, execute_queue
from projeto_rl.runner import load_run, run_experiment
from projeto_rl.utils.io import ensure_dir, load_json, save_json


def _print_plan(console, plan: dict) -> None:
    console.title("Training Plan", "Offline optimization, then evaluation of the selected checkpoint in the simulator.")
    console.table(
        "Run",
        ("field", "value"),
        [
            (key, plan[key])
            for key in ("run_id", "output_dir", "dataset_id", "environment", "method_id", "representation", "seed")
        ],
    )
    console.table("Budget", ("field", "value"), plan["budget"].items())
    console.table("Model selection", ("field", "value"), plan["model_selection"].items())
    if not plan["preflight"]["ready"]:
        console.warning(f"Dataset is incomplete: {plan['preflight']['missing']}. Build it before training.")


def train(args) -> int:
    console = common.console(args)
    run = common.prepare_run(args, dataset_id=args.dataset_id, method_id=args.method_id, seed=args.seed)
    plan = common.run_plan(run)
    if args.dry_run:
        if args.json:
            console.emit(plan)
        else:
            _print_plan(console, plan)
        return 0
    if not args.json:
        _print_plan(console, plan)
    if not run["preflight"]["ready"]:
        raise FileNotFoundError(
            f"Dataset {args.dataset_id!r} is not ready. Run: projeto-rl dataset build {args.dataset_id}"
        )
    with common.device(args.device):
        output = run_experiment(
            run["params"], output_dir=run["output_dir"], manifest=run["manifest"], force=args.force, quiet=True
        )
    if args.json:
        console.emit({**plan, "status": "completed", "output_dir": str(output)})
    else:
        console.success(f"Completed {run['manifest']['run_id']}")
        console.note(f"Artifacts: {output}")
    return 0


def queue_jobs(
    args,
    *,
    dataset_ids: list[str],
    methods: list[str] | None,
    seeds: list[int],
    representations: list[str] | None = None,
) -> list[QueueJob]:
    """One job per dataset x method x representation x seed; ``None`` keeps the configured representation."""

    bench = common.benchmark(args)
    jobs = []
    for dataset_id in dataset_ids:
        for method_id in methods or [method.method_id for method in bench.methods(dataset_id)]:
            for representation in representations or [None]:
                for seed in seeds:
                    run = common.prepare_run(
                        args, dataset_id=dataset_id, method_id=method_id, seed=seed, representation=representation
                    )
                    params = run["params"]
                    jobs.append(
                        QueueJob(
                            name=f"{dataset_id}:{method_id}:{params['representation']}:seed-{seed}",
                            params=params,
                            output_dir=str(run["output_dir"]),
                            manifest=run["manifest"],
                            device=args.device,
                            force=args.force,
                            ready=run["preflight"]["ready"],
                        )
                    )
    return jobs


def run_queue(args, jobs: list[QueueJob], *, title: str) -> int:
    console = common.console(args)
    if args.dry_run or not args.execute:
        payload = {"jobs": [{"name": job.name, "ready": job.ready, "output_dir": job.output_dir} for job in jobs]}
        if args.json:
            console.emit(payload)
        else:
            console.title(title, "Add --execute after reviewing the resolved jobs.")
            console.table(
                "Jobs", ("name", "data ready", "output"), [(job.name, job.ready, job.output_dir) for job in jobs]
            )
        return 0
    not_ready = sorted({job.params["dataset_id"] for job in jobs if not job.ready})
    if not_ready:
        raise FileNotFoundError(f"Datasets are incomplete: {not_ready}. Run `projeto-rl dataset build` first.")
    queue_dir, manifest = execute_queue(
        jobs,
        root=common.root(args),
        max_workers=args.jobs,
        gpus=args.gpus,
        resume=not args.force,
        follow=bool(args.follow and not args.json),
    )
    if args.json:
        console.emit(
            {"queue_dir": str(queue_dir), "suite_results": str(queue_dir / "suite_results.json"), "manifest": manifest}
        )
    else:
        console.success(f"Queue finished: {queue_dir}")
        console.table(
            "Jobs",
            ("name", "status", "log"),
            [(item["name"], item["status"], item["log_path"]) for item in manifest["jobs"]],
        )
        console.note(f"Results: {queue_dir / 'suite_results.json'}")
    return 0 if not manifest["failures"] else 1


def suite(args) -> int:
    jobs = queue_jobs(args, dataset_ids=[args.dataset_id], methods=args.methods, seeds=common.parse_seeds(args.seeds))
    return run_queue(args, jobs, title=f"Suite: {args.dataset_id}")


def _run_summary(run_dir, manifest) -> dict:
    return {
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id"),
        "status": effective_status(manifest),
        "dataset_id": manifest.get("dataset_id"),
        "method_id": manifest.get("method_id"),
        "representation": manifest.get("representation"),
        "seed": manifest.get("seed"),
        "updated_at": manifest.get("updated_at"),
    }


def status(args) -> int:
    console = common.console(args)
    if args.run:
        run_dir = resolve_run(args.run, common.root(args))
        payload = {
            **_run_summary(run_dir, load_manifest(run_dir)),
            "artifacts": sorted(path.name for path in run_dir.iterdir() if path.is_file()),
        }
        if args.json:
            console.emit(payload)
        else:
            console.table("Run status", ("field", "value"), payload.items())
        return 0
    runs = [_run_summary(path, manifest) for path, manifest in discover_runs(common.root(args))[: args.limit]]
    if args.json:
        console.emit({"runs": runs})
    else:
        console.title("Run Status", f"Most recent {len(runs)} runs ('stale' = marked running but its process is gone)")
        console.table(
            "Runs",
            ("status", "dataset", "method", "representation", "seed", "updated", "directory"),
            [
                (
                    item["status"],
                    item["dataset_id"],
                    item["method_id"],
                    item["representation"],
                    item["seed"],
                    item["updated_at"],
                    item["run_dir"],
                )
                for item in runs
            ],
        )
    return 0


def inspect_run(args) -> int:
    console = common.console(args)
    run_dir = resolve_run(args.run, common.root(args))
    metrics = load_json(run_dir / "metrics_summary.json") if (run_dir / "metrics_summary.json").exists() else {}
    payload = {
        **_run_summary(run_dir, load_manifest(run_dir)),
        "optimization": metrics.get("train_summary", {}),
        "direct_evaluation": load_json(run_dir / "eval_summary.json")
        if (run_dir / "eval_summary.json").exists()
        else {},
    }
    if args.json:
        console.emit(payload)
        return 0
    console.title("Run Inspection", payload["run_id"])
    console.table("Optimization", ("metric", "value"), payload["optimization"].items())
    console.table("Direct evaluation", ("metric", "value"), payload["direct_evaluation"].items())
    return 0


def evaluate(args) -> int:
    console = common.console(args)
    run_dir = resolve_run(args.run, common.root(args))
    config, policy = load_run(run_dir)
    destination = ensure_dir(run_dir / "evaluations" / common.timestamp())
    with common.device(args.device):
        result = evaluate_policy(
            policy,
            config=config,
            num_episodes=args.episodes or int(config["eval_episodes"]),
            eval_seed_base=args.seed_base if args.seed_base is not None else int(config["eval_seed_base"]),
            output_dir=destination,
            label="re-evaluated",
            show_progress=not args.no_progress,
        )
    save_json(result["summary"], destination / "eval_summary.json")
    if args.json:
        console.emit({"run": str(run_dir), "evaluation_dir": str(destination), "summary": result["summary"]})
    else:
        console.success(f"Evaluation written to {destination}")
        console.table("Direct simulator evaluation", ("metric", "value"), result["summary"].items())
    return 0


def report(args) -> int:
    console = common.console(args)
    destination, summary = build_run_report(resolve_run(args.run, common.root(args)), root=common.root(args))
    if args.json:
        console.emit({"report_dir": str(destination), "summary": summary})
    else:
        console.success(f"Report written to {destination}")
    return 0


def register(subparsers) -> None:
    parser = subparsers.add_parser("train", help="Train and evaluate one dataset/method/seed.")
    parser.add_argument("dataset_id")
    parser.add_argument("method_id")
    common.add_run_options(parser)
    parser.set_defaults(handler=train)

    parser = subparsers.add_parser("suite", help="Queue every (or selected) method on one dataset over seeds.")
    parser.add_argument("dataset_id")
    parser.add_argument("--method", dest="methods", action="append", help="Limit to a method; repeatable.")
    common.add_queue_options(parser, default_seeds="0")
    common.add_run_options(parser)
    parser.set_defaults(handler=suite)

    parser = subparsers.add_parser("status", help="Show run status.")
    parser.add_argument("run", nargs="?")
    parser.add_argument("--limit", type=int, default=20)
    parser.set_defaults(handler=status)

    inspect = subparsers.add_parser("inspect", help="Inspect a dataset or a run.").add_subparsers(
        dest="inspect_command", required=True
    )
    item = inspect.add_parser("dataset")
    item.add_argument("dataset_id")
    item.set_defaults(handler=inspect_dataset)
    item = inspect.add_parser("run")
    item.add_argument("run")
    item.set_defaults(handler=inspect_run)

    parser = subparsers.add_parser("evaluate", help="Fresh simulator evaluation of a completed run.")
    parser.add_argument("run")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.set_defaults(handler=evaluate)

    parser = subparsers.add_parser("report", help="Write a Markdown/JSON report for one run.")
    parser.add_argument("run")
    parser.set_defaults(handler=report)
