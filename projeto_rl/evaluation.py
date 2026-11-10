"""Direct simulator evaluation of a policy, with trajectory exports."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from projeto_rl.envs import make_env
from projeto_rl.utils.io import ensure_dir, save_json
from projeto_rl.utils.metrics import action_stability
from projeto_rl.utils.progress import tqdm
from projeto_rl.utils.queue_progress import update_progress


def _mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or frame.empty:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return float(values.mean()) if values.notna().any() else 0.0


def evaluate_policy(
    policy,
    *,
    config: dict[str, Any],
    num_episodes: int,
    eval_seed_base: int,
    output_dir: str | Path | None = None,
    label: str = "best",
    export_csv: bool = True,
    show_progress: bool = False,
    deterministic: bool | None = None,
) -> dict[str, Any]:
    """Roll out ``policy`` for ``num_episodes`` seeded episodes and summarize them.

    The summary always contains the common fields (``mean_return``,
    ``mean_success``, ``mean_length``, action-change statistics, and latency),
    ``mean_<metric>`` for every per-step metric declared by the
    environment, and ``outcome_<name>`` for every extra episode outcome.
    """

    deterministic = bool(config.get("eval_deterministic", True) if deterministic is None else deterministic)
    env = make_env(config, purpose="evaluate")
    policy.bind_env(env)
    method = str(config.get("method_id", config.get("algo", "policy")))
    trajectories: list[dict] = []
    episodes: list[dict] = []
    latencies: list[float] = []
    progress_every = max(1, int(num_episodes) // 20)
    update_progress(status="running", phase="evaluation", method_id=method, step=0, total_steps=int(num_episodes))
    for episode_id in tqdm(
        range(int(num_episodes)), desc=f"eval {label}", unit="ep", leave=False, disable=not show_progress
    ):
        policy.reset_episode()
        obs, info = env.reset(seed=int(eval_seed_base) + episode_id)
        initial_state = int(info["state_id"])
        metric_rows = [env.transition_metrics(info)]
        actions: list[int] = []
        total_reward = 0.0
        first_row = len(trajectories)
        done = False
        while not done:
            started = time.perf_counter()
            action = int(policy.act(obs, info, deterministic=deterministic))
            latencies.append(time.perf_counter() - started)
            if np.asarray(info["action_mask"])[action] <= 0:
                raise ValueError(
                    f"Policy {method!r} emitted inadmissible action {action}; admissible={info['admissible_actions']}"
                )
            current_metrics = env.transition_metrics(info)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            policy.observe(action, reward)
            done = bool(terminated or truncated)
            trajectories.append(
                {
                    "episode_id": episode_id,
                    "eval_seed": int(eval_seed_base) + episode_id,
                    "t": len(actions),
                    "state_id": int(info["state_id"]),
                    "action": action,
                    "action_label": env.action_label(action),
                    **env.export_columns(info, action),
                    "reward": float(reward),
                    "done": done,
                    **{f"metric_{name}": float(value) for name, value in current_metrics.items()},
                    **{
                        f"feature_{name}": float(value)
                        for name, value in zip(env.spec.feature_names, np.asarray(obs, dtype=np.float32))
                    },
                }
            )
            actions.append(action)
            total_reward += float(reward)
            obs, info = next_obs, next_info
            metric_rows.append(env.transition_metrics(info))

        outcome = env.episode_metrics(total_reward=total_reward, final_info=info, transition_metrics=metric_rows)
        stability_l1, abrupt_rate = action_stability(actions, env.action_distance, abrupt_fn=env.action_is_abrupt)
        episode = {
            "episode_id": episode_id,
            "eval_seed": int(eval_seed_base) + episode_id,
            "return": float(total_reward),
            "success": bool(outcome["success"]),
            "length": len(actions),
            "initial_state": initial_state,
            "action_stability_l1": stability_l1,
            "abrupt_jump_rate": abrupt_rate,
            **{f"outcome_{name}": value for name, value in outcome.items() if name != "success"},
        }
        for name in env.spec.metric_names:
            values = [row[name] for row in metric_rows]
            episode[f"initial_{name}"] = float(values[0])
            episode[f"mean_{name}"] = float(np.mean(values))
            episode[f"final_{name}"] = float(values[-1])
        episodes.append(episode)
        for row in trajectories[first_row:]:
            row["episode_success"] = episode["success"]
        if episode_id == 0 or (episode_id + 1) % progress_every == 0 or episode_id + 1 == int(num_episodes):
            update_progress(
                status="running",
                phase="evaluation",
                method_id=method,
                step=episode_id + 1,
                total_steps=int(num_episodes),
            )

    env.close()
    episodes_df = pd.DataFrame(episodes)
    trajectories_df = pd.DataFrame(trajectories)
    summary: dict[str, Any] = {
        "episodes": int(len(episodes_df)),
        "mean_return": _mean(episodes_df, "return"),
        "mean_success": _mean(episodes_df, "success"),
        "mean_length": _mean(episodes_df, "length"),
        "mean_action_stability_l1": _mean(episodes_df, "action_stability_l1"),
        "mean_abrupt_jump_rate": _mean(episodes_df, "abrupt_jump_rate"),
        "mean_action_latency_ms": float(np.mean(latencies) * 1000.0) if latencies else 0.0,
        **{f"mean_{name}": _mean(episodes_df, f"mean_{name}") for name in env.spec.metric_names},
        **{column: _mean(episodes_df, column) for column in episodes_df.columns if column.startswith("outcome_")},
    }
    if output_dir is not None:
        output_dir = ensure_dir(output_dir)
        save_json(env.spec.to_dict(), output_dir / "environment_spec.json")
        if export_csv:
            episodes_df.to_csv(output_dir / "episodes_summary.csv", index=False)
            trajectories_df.to_csv(output_dir / "trajectories.csv", index=False)
    return {
        "episodes": episodes_df,
        "trajectories": trajectories_df,
        "summary": summary,
        "environment": env.spec.to_dict(),
    }
