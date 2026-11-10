# Algorithms

| Method id | Summary |
|---|---|
| `bc` | Behavior cloning: masked cross-entropy on logged actions |
| `dqn` | Offline DQN / fitted Q-iteration with a soft-updated target network |
| `iql` | Discrete IQL (Kostrikov et al., 2022): expectile value, twin critics, advantage-weighted actor |
| `cql` | Discrete CQL (Kumar et al., 2020): TD regression plus a conservative logsumexp penalty |

All methods are MLPs over `float32` observation vectors (after the run's
representation) and respect the action mask: Q-values and logits of
inadmissible actions are set to `-1e9`. Their hyperparameters are in
`configs/algorithms/<algo>.json`; nothing was tuned for this project.

## The algorithm interface

Each method is one module in `projeto_rl/algorithms/` defining an
`OfflineAlgorithm` subclass (`projeto_rl/algorithms/base.py`):

```python
@register_algorithm
class CQL(OfflineAlgorithm):
    name = "cql"  # `algo` in configs
    method_id = "cql"  # public id
    validation_objective = "validation conservative Q objective"

    def build(self): ...  # networks; the returned module is checkpointed
    def loss(self, batch): ...  # (loss tensor, dict of floats)
    def make_policy(self): ...  # deployed Policy: act() and action_probabilities()
```

Optional hooks: `after_update` (target networks), `batches`, `update`,
`validate`, `state_dict`/`from_checkpoint`. The shared loop in
`projeto_rl/training.py` does representation fitting, batching, validation every
`offline_log_interval` steps, early stopping, best-checkpoint selection,
gradient clipping (`max_grad_norm`), and logging. An algorithm sees only the
`DataSpec` (input size after the representation, number of actions) and
batches.

`load_policy(config, checkpoint)` rebuilds the deployed policy of any run,
including its frozen representation.

## Method notes

**TD targets.** DQN, IQL, and CQL share `td_target`
(`projeto_rl/algorithms/base.py`): `r + γ · V(s') · (1 − terminal)`. Only true
terminal states stop the bootstrap; a step cut by a time limit (`timeouts`)
still bootstraps. Rewards arrive divided by the train split's mean `|reward|`
([experiments.md](experiments.md)).

**DQN.** One-step masked TD targets from a target network updated every
`target_network_frequency` steps with rate `tau`. Deployment is greedy.

**IQL / CQL.** Twin masked Q-networks. IQL fits an expectile value and extracts
the actor by advantage-weighted regression; CQL adds
`logsumexp(Q) - Q(logged action)` over admissible actions. CQL deploys the
minimum of its twin Q-networks; IQL deploys its actor.
