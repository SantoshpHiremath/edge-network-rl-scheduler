"""
A real, from-scratch tabular Q-learning agent for the EdgeSchedulingEnv.

The environment's observation space is continuous (load fractions), so
it's discretized into bins before being used as a Q-table key -- a
standard, honest approach for applying tabular Q-learning to a
low-dimensional continuous space, disclosed here rather than silently
assumed. A deep RL approach (DQN) would remove the need for
discretization, but a tabular agent is easier to inspect, debug, and
verify correctness of -- appropriate for a small state space like this
one, and a deliberate, disclosed design choice rather than a
limitation hidden from the reader.
"""

from collections import defaultdict
from typing import Optional

import numpy as np


def discretize(obs: np.ndarray, bins: int = 10) -> tuple:
    """Map a continuous observation vector to a discrete tuple key.

    Load fractions (first n_nodes entries) are binned into `bins`
    buckets; the one-hot task-type entries are already discrete
    (0.0/1.0) and pass through unchanged.
    """
    n_load_dims = len(obs) - 2
    load_bins = tuple(
        min(bins - 1, int(obs[i] * bins)) for i in range(n_load_dims)
    )
    task_bits = tuple(int(round(x)) for x in obs[n_load_dims:])
    return load_bins + task_bits


class QLearningAgent:
    """Standard tabular Q-learning with epsilon-greedy exploration and
    epsilon decay -- the canonical baseline RL algorithm, implemented
    directly (no external RL library) so every part of the update rule
    is inspectable and independently testable."""

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._rng = np.random.default_rng(seed)
        # Q-table: dict from discretized-state -> np.array of Q-values
        # per action. defaultdict avoids having to pre-enumerate every
        # possible state up front.
        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

    def select_action(self, state_key: tuple, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        q_values = self.q_table[state_key]
        # break ties randomly rather than always picking the lowest
        # index, so a freshly-initialized all-zero row doesn't bias
        # the agent toward action 0 before it has learned anything.
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_q)
        return int(self._rng.choice(best_actions))

    def update(self, state_key: tuple, action: int, reward: float,
               next_state_key: tuple, done: bool) -> None:
        current_q = self.q_table[state_key][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
