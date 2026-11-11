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


def _episode(env, policy, *, episode_id: int, seed: int, method: str):
    """One episode as a coroutine: yields ``(obs, info)``, is sent the action, returns ``(episode, rows)``."""

    policy.reset_episode()
    obs, info = env.reset(seed=seed)
    initial_state = int(info["state_id"])
    metric_rows = [env.transition_metrics(info)]
    actions: list[int] = []
    rows: list[dict] = []
    total_reward = 0.0
    done = False
    while not done:
        action = yield obs, info
        if np.asarray(info["action_mask"])[action] <= 0:
            raise ValueError(
                f"Policy {method!r} emitted inadmissible action {action}; admissible={info['admissible_actions']}"
            )
        current_metrics = env.transition_metrics(info)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        policy.observe(action, reward)
        done = bool(terminated or truncated)
        rows.append(
            {
                "episode_id": episode_id,
                "eval_seed": seed,
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
        "eval_seed": seed,
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
    for row in rows:
        row["episode_success"] = episode["success"]
    return episode, rows


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

    A stateless policy acting greedily runs ``eval_parallel_episodes``
    episodes side by side, each in its own environment, and chooses their
    actions in one :meth:`Policy.act_batch` call per step. Every episode is
    still seeded by its own index, so the episodes are the ones a sequential
    evaluation produces; the latency is then the time of the call divided by
    the number of episodes in it. Any other policy runs one episode at a time.
    """

    num_episodes = int(num_episodes)
    deterministic = bool(config.get("eval_deterministic", True) if deterministic is None else deterministic)
    # Side by side only when the order of the calls cannot matter: no state
    # between calls and no random draws from the policy's generator.
    parallel = int(config.get("eval_parallel_episodes", 1)) if policy.stateless and deterministic else 1
    if parallel <= 0:
        raise ValueError("eval_parallel_episodes must be > 0")
    envs = [make_env(config, purpose="evaluate") for _ in range(max(1, min(parallel, num_episodes)))]
    env = envs[0]
    policy.bind_env(env)
    method = str(config.get("method_id", config.get("algo", "policy")))
    finished: dict[int, tuple[dict, list[dict]]] = {}
    latencies: list[float] = []
    progress_every = max(1, num_episodes // 20)
    update_progress(status="running", phase="evaluation", method_id=method, step=0, total_steps=num_episodes)
    bar = tqdm(total=num_episodes, desc=f"eval {label}", unit="ep", leave=False, disable=not show_progress)
    waiting = iter(range(num_episodes))

    def start(slot_env):
        episode_id = next(waiting, None)
        if episode_id is None:
            return None
        run = _episode(slot_env, policy, episode_id=episode_id, seed=int(eval_seed_base) + episode_id, method=method)
        return slot_env, run, next(run)

    running = [slot for slot in map(start, envs) if slot is not None]
    while running:
        started = time.perf_counter()
        actions = policy.act_batch(
            [obs for _, _, (obs, _) in running], [info for _, _, (_, info) in running], deterministic=deterministic
        )
        latencies.extend([(time.perf_counter() - started) / len(running)] * len(running))
        advanced = []
        for (slot_env, run, _), action in zip(running, actions, strict=True):
            try:
                advanced.append((slot_env, run, run.send(int(action))))
                continue
            except StopIteration as stop:
                episode, rows = stop.value
            finished[episode["episode_id"]] = (episode, rows)
            bar.update(1)
            if len(finished) == 1 or len(finished) % progress_every == 0 or len(finished) == num_episodes:
                update_progress(
                    status="running",
                    phase="evaluation",
                    method_id=method,
                    step=len(finished),
                    total_steps=num_episodes,
                )
            slot = start(slot_env)
            if slot is not None:
                advanced.append(slot)
        running = advanced

    bar.close()
    for slot_env in envs:
        slot_env.close()
    episodes = [finished[episode_id][0] for episode_id in range(num_episodes)]
    trajectories = [row for episode_id in range(num_episodes) for row in finished[episode_id][1]]
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
