# Data Schema (version 5)

Every split (`train.npz`, `val.npz`) stores these arrays, aligned on the first
axis `N` (one row per logged transition). Observations are always raw;
representations are applied per run, in memory.

| Array | dtype | Shape | Meaning |
|---|---|---|---|
| `observations`, `next_observations` | float32 | `(N, obs_dim)` | Raw observation before and after the step |
| `actions` | int64 | `(N,)` | Logged action |
| `next_actions` | int64 | `(N,)` | Next logged action in the episode, `-1` at the end |
| `rewards` | float32 | `(N,)` | Environment reward |
| `terminals` | bool | `(N,)` | Episode reached a true terminal state (`terminated`); the TD target does not bootstrap |
| `timeouts` | bool | `(N,)` | Episode was cut by a time limit (`truncated` and not `terminated`); the TD target still bootstraps |
| `action_masks`, `next_action_masks` | float32 | `(N, num_actions)` | Admissible actions (1) before and after the step |
| `behavior_logprobs` | float32 | `(N,)` | Exact log-probability of the logged action under the behavior policy (with its ε) |
| `behavior_policy_ids` | int64 | `(N,)` | Stable behavior-policy id (declared by the environment) |
| `episode_ids`, `timesteps` | int64 | `(N,)` | Trajectory reconstruction keys |
| `state_ids`, `next_state_ids` | int64 | `(N,)` | Simulator diagnostics, never agent inputs |
| `env_metrics`, `next_env_metrics` | float32 | `(N, num_metrics)` | Per-step metrics in the order of `metadata.environment.metric_names` |

`metadata.json` records the collection config, the environment spec,
`observation.features` (name, kind, and cardinality of every column, read by
the representations), the behavior policy with its id and `epsilon`,
per-split statistics, and SHA-256 hashes of the split files.

## How methods consume the data

| Consumer | Uses |
|---|---|
| representation `fit` | the train split (observations, and any other array it needs) |
| BC | observation, action mask, action |
| DQN, CQL | observation, action, reward, next observation, next mask, terminal |
| IQL | the DQN fields; its value network bootstraps from the next observation |
