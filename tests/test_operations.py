"""CLI, run manifests, queue progress, the summary statistics, and run-all."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys

import pytest

from projeto_rl.cli import main
from projeto_rl.operations.artifacts import effective_status, initial_manifest, run_directory, write_manifest
from projeto_rl.operations.baselines import baseline_path
from projeto_rl.operations.queue import _write_suite_results, estimate_remaining_seconds
from projeto_rl.operations.summary import (
    interquartile_mean,
    normalized_score,
    paired_difference,
    summarize,
    summarize_runs,
)
from projeto_rl.paths import CONFIG_ROOT
from projeto_rl.utils.io import save_json
from projeto_rl.utils.progress import format_seconds


def _cli(capsys, *argv) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_train_dry_run_emits_one_json_plan(capsys, tmp_path):
    code, out, _ = _cli(
        capsys, "train", "epicare/soc/dev", "dqn", "--artifact-root", str(tmp_path), "--dry-run", "--json"
    )
    plan = json.loads(out)
    assert code == 0 and plan["dataset_id"] == "epicare/soc/dev" and plan["representation"] == "standardize"
    assert plan["preflight"]["ready"] is False and plan["model_selection"]["meaning"] == "validation TD loss"


def test_unknown_method_is_a_clean_error(capsys):
    code, _, err = _cli(capsys, "train", "epicare/soc/dev", "ppo", "--dry-run")
    assert code == 2 and "Available:" in err and "Traceback" not in err


def test_module_entry_point():
    command = [sys.executable, "-m", "projeto_rl", "catalog", "--json"]
    payload = json.loads(subprocess.run(command, text=True, capture_output=True, check=True).stdout)
    assert len(payload["datasets"]) == 8


def test_manifest_lifecycle_and_stale_detection(tmp_path):
    params = {
        "dataset_id": "epicare/soc/dev",
        "env_id": "epicare",
        "algo": "dqn",
        "method_id": "dqn",
        "representation": "standardize",
        "seed": 4,
    }
    run_dir = run_directory(params, root=tmp_path)
    assert run_dir.relative_to(tmp_path).parts[:6] == ("runs", "epicare", "soc", "dev", "dqn", "standardize")
    manifest = write_manifest(run_dir, initial_manifest(params, run_dir=run_dir), status="running")
    assert ":standardize:seed-4:" in manifest["run_id"]
    assert effective_status(manifest) == "running"  # this process is alive
    manifest["process"]["pid"] = 2**22 + 12345  # no such process
    assert effective_status(manifest) == "stale"
    assert write_manifest(run_dir, manifest, status="interrupted")["finished_at"]


def test_normalized_score_by_hand():
    assert normalized_score(5.0, low=1.0, high=9.0) == pytest.approx(50.0)
    assert normalized_score(13.0, low=1.0, high=9.0) == pytest.approx(150.0)
    assert normalized_score(-70.0, low=-80.0, high=-60.0) == pytest.approx(50.0)
    with pytest.raises(ValueError, match="equal"):
        normalized_score(1.0, low=2.0, high=2.0)


def test_paired_difference_by_hand():
    result = paired_difference({0: 11.0, 1: 22.0, 2: 33.0}, {0: 10.0, 1: 20.0, 2: 30.0})
    # Differences 1, 2, 3: mean 2, sample std 1, t(0.975, df=2) = 4.302652729911275.
    half_width = 4.302652729911275 / math.sqrt(3)
    assert result["mean_difference"] == pytest.approx(2.0)
    assert (result["ci95_low"], result["ci95_high"]) == pytest.approx((2.0 - half_width, 2.0 + half_width))
    assert (result["wins"], result["num_seeds"]) == (3, 3)
    single = paired_difference({0: 5.0}, {0: 7.0})
    assert single["mean_difference"] == -2.0 and math.isnan(single["ci95_low"]) and single["wins"] == 0
    with pytest.raises(ValueError, match="Unpaired"):
        paired_difference({0: 1.0, 1: 2.0}, {0: 1.0})


def test_interquartile_mean_trims_a_quarter_from_each_end():
    assert interquartile_mean([1.0, 2.0, 3.0, 100.0]) == pytest.approx(2.5)  # mean of 2 and 3
    assert interquartile_mean([-500.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 900.0]) == pytest.approx(35.0)
    assert interquartile_mean([4.0]) == 4.0


def test_summary_groups_scores_and_compares_against_the_reference():
    rows = [
        {
            "dataset_id": "a/x/dev",
            "env_id": "a",
            "method_id": "bc",
            "representation": rep,
            "seed": seed,
            "mean_return": value,
            "mean_success": 0.0,
            "optimization_seconds": 1.0,
            "evaluation_seconds": 1.0,
        }
        for rep, seed, value in (("raw", 0, 2.0), ("raw", 1, 4.0), ("standardize", 0, 6.0), ("standardize", 1, 6.0))
    ]
    summary = summarize(
        rows, references={"a/x/dev": (0.0, 10.0)}, groups={"a": "heterogeneous"}, reference_representation="standardize"
    )
    assert [row["normalized_score"] for row in summary["rows"]] == [20.0, 40.0, 60.0, 60.0]
    (comparison,) = summary["comparison"]
    assert comparison["representation"] == "raw" and comparison["mean_difference"] == pytest.approx(-30.0)
    aggregate = {(item["group"], item["representation"]): item["iqm_normalized_score"] for item in summary["aggregate"]}
    assert aggregate[("heterogeneous", "raw")] == pytest.approx(30.0)
    assert aggregate[("all", "standardize")] == pytest.approx(60.0)


def _fake_run(root, representation, mean_return, seed=0):
    params = {
        "dataset_id": "epicare/soc/dev",
        "env_id": "epicare",
        "algo": "dqn",
        "method_id": "dqn",
        "representation": representation,
        "seed": seed,
    }
    run_dir = run_directory(params, root=root)
    write_manifest(run_dir, initial_manifest(params, run_dir=run_dir), status="completed")
    save_json(
        {
            "eval_summary": {"mean_return": mean_return, "mean_success": 0.5},
            "timing_summary": {"optimization_seconds": 2.0, "evaluation_seconds": 1.0},
            "train_summary": {"best_validation_step": 2},
        },
        run_dir / "metrics_summary.json",
    )
    (run_dir / "train_history.csv").write_text("step,train_loss,validation_loss\n1,3.0,2.5\n2,2.0,1.5\n3,1.0,1.8\n")
    return run_dir


def test_summarize_runs_reads_baselines_and_writes_reports(tmp_path):
    experiment = {"representations": ["standardize", "raw"], "reference_representation": "standardize"}
    run_dirs = [_fake_run(tmp_path, "raw", 3.0), _fake_run(tmp_path, "standardize", 5.0)]
    unfinished = _fake_run(tmp_path, "raw", 3.0, seed=9)
    write_manifest(unfinished, json.loads((unfinished / "run_manifest.json").read_text()), status="running")
    with pytest.raises(ValueError, match="Only 2 of the experiment's 3 runs"):
        summarize_runs(tmp_path, run_dirs=[*run_dirs, unfinished], experiment=experiment)
    with pytest.raises(FileNotFoundError, match="projeto-rl baselines"):
        summarize_runs(tmp_path, run_dirs=run_dirs, experiment=experiment)
    save_json(
        {"references": {"low": {"mean_return": 1.0}, "high": {"mean_return": 5.0}}},
        baseline_path("epicare/soc/dev", tmp_path),
    )
    payload = summarize_runs(tmp_path, run_dirs=run_dirs, experiment=experiment, output_dir=tmp_path / "out")
    (comparison,) = payload["comparison"]
    assert comparison["representation"] == "raw" and comparison["mean_difference"] == pytest.approx(-50.0)
    for name in ("runs.csv", "by_method.csv", "comparison.csv", "summary.json", "summary.md"):
        assert (tmp_path / "out" / name).exists()
    assert payload["loss_plots"] == [str(tmp_path / "out" / "loss_curves" / "epicare__soc__dev.png")]
    assert (tmp_path / "out" / "loss_curves" / "epicare__soc__dev.png").stat().st_size > 0
    assert "loss_curves/epicare__soc__dev.png" in (tmp_path / "out" / "summary.md").read_text()


def test_suite_results_count_jobs(tmp_path):
    run_dir = _fake_run(tmp_path, "raw", 3.0)
    queue_dir = tmp_path / "queues" / "q"
    queue_dir.mkdir(parents=True)
    jobs = [{"name": "a", "status": "completed", "output_dir": str(run_dir), "method_id": "dqn"}]
    payload = json.loads(
        _write_suite_results(
            queue_dir, {"queue_id": "q", "finished_at": "now", "failures": 0, "jobs": jobs}
        ).read_text()
    )
    assert payload["suite"]["status"] == "completed" and payload["counts"]["completed"] == 1
    assert payload["jobs"][0]["representation"] == "raw"


def test_eta_uses_per_method_durations_shared_by_workers():
    records = [
        {"method_id": "bc", "status": "completed", "duration_seconds": 10.0},
        {"method_id": "bc", "status": "completed", "duration_seconds": 20.0},
        {"method_id": "cql", "status": "completed", "duration_seconds": 60.0},
        {"method_id": "bc", "status": "pending"},  # 15 expected
        {"method_id": "cql", "status": "pending"},  # 60 expected
        {"method_id": "iql", "status": "pending"},  # no iql finished: overall mean 30
        {"method_id": "cql", "status": "running", "started_monotonic": 100.0},  # 60 - 40 elapsed = 20
        {"method_id": "bc", "status": "skipped"},
    ]
    assert estimate_remaining_seconds(records, workers=5, now=140.0) == pytest.approx((15 + 60 + 30 + 20) / 5)
    assert estimate_remaining_seconds([{"method_id": "bc", "status": "pending"}], workers=1, now=0.0) is None


def test_format_seconds():
    assert format_seconds(9) == "9s" and format_seconds(65) == "1m05s" and format_seconds(3725) == "1h02m05s"


def _tiny_config_root(tmp_path):
    root = tmp_path / "configs"
    shutil.copytree(CONFIG_ROOT, root)
    protocol = json.loads((root / "protocol.json").read_text())
    protocol.update({"offline_train_steps": 4, "offline_log_interval": 2, "offline_batch_size": 8})
    protocol["tiers"]["dev"]["eval_episodes"] = 2
    (root / "protocol.json").write_text(json.dumps(protocol))
    for env_id in ("epicare", "industrial_benchmark"):
        spec = json.loads((root / "datasets" / f"{env_id}.json").read_text())
        spec["tiers"]["dev"] = {"num_train_episodes": 4, "num_val_episodes": 2}
        (root / "datasets" / f"{env_id}.json").write_text(json.dumps(spec))
    env = json.loads((root / "envs" / "industrial_benchmark.json").read_text())
    env["run"]["env_kwargs"]["horizon"] = 10
    (root / "envs" / "industrial_benchmark.json").write_text(json.dumps(env))
    experiment = {
        "datasets": ["epicare/soc/dev", "industrial-benchmark/mediocre/dev"],
        "methods": ["bc"],
        "representations": ["raw", "standardize"],
        "reference_representation": "standardize",
        "seeds": [0],
        "progress_interval_seconds": 1,
    }
    (root / "experiments" / "tiny.json").write_text(json.dumps(experiment))
    return root


@pytest.mark.slow
def test_run_all_end_to_end_and_resume(capsys, tmp_path):
    root, artifacts = _tiny_config_root(tmp_path), tmp_path / "artifacts"
    argv = [
        "--config-root",
        str(root),
        "--artifact-root",
        str(artifacts),
        "run-all",
        str(root / "experiments" / "tiny.json"),
    ]
    code, _, err = _cli(capsys, *argv, "--jobs", "2", "--device", "cpu")
    assert code == 0, err
    assert "[4/4 summary] done" in err and "[3/4 training] 4/4 runs done" in err
    summaries = sorted((artifacts / "reports" / "summary").iterdir())
    assert len(summaries) == 1 and (summaries[0] / "summary.md").exists()
    built = (artifacts / "datasets" / "epicare" / "soc" / "dev" / "train.npz").stat().st_mtime_ns
    code, out, _ = _cli(capsys, *argv, "--dry-run", "--json")
    plan = json.loads(out)
    assert (
        code == 0 and plan["runs_pending"] == 0 and not plan["datasets_to_build"] and not plan["baselines_to_evaluate"]
    )
    code, _, err = _cli(capsys, *argv)
    assert code == 0 and "already built" in err and "baselines up to date" in err
    assert (artifacts / "datasets" / "epicare" / "soc" / "dev" / "train.npz").stat().st_mtime_ns == built
