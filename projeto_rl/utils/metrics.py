"""Episode-level action statistics."""

from __future__ import annotations

import numpy as np


def action_stability(actions: list[int], distance_fn, *, abrupt_fn) -> tuple[float, float]:
    """Mean action-change distance and rate of abrupt changes within one episode."""

    if len(actions) <= 1:
        return 0.0, 0.0
    pairs = list(zip(actions, actions[1:]))
    deltas = [float(distance_fn(previous, current)) for previous, current in pairs]
    abrupt = [float(bool(abrupt_fn(previous, current))) for previous, current in pairs]
    return float(np.mean(deltas)), float(np.mean(abrupt))
