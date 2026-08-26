"""
A custom Gymnasium environment modeling task scheduling across a small
industrial edge network: N edge nodes with limited compute capacity and
per-node latency, and a stream of incoming tasks that must be assigned
to a node (or dropped) as they arrive.

This is deliberately scoped small and discrete (a handful of nodes, a
handful of task types) so a from-scratch Q-learning agent can actually
learn a good policy in a reasonable number of episodes on a CPU, while
still being a genuine sequential decision problem with real tradeoffs:
sending every task to the fastest node overloads it and causes drops,
while spreading tasks evenly can miss the fact that some tasks are
latency-critical and some nodes are meaningfully faster.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass(frozen=True)
class EdgeNode:
    """A single edge node's static characteristics."""

    name: str
    capacity: int  # max concurrent task "load units" it can hold
    speed: float  # higher = processes load faster (load units cleared per step)


# Three nodes with deliberately different profiles, not just scaled
# copies of each other -- this is what makes the routing decision
# genuinely non-trivial rather than "always pick the biggest number".
DEFAULT_NODES = (
    EdgeNode(name="edge-fast-small", capacity=3, speed=2.0),
    EdgeNode(name="edge-slow-large", capacity=8, speed=0.75),
    EdgeNode(name="edge-balanced", capacity=5, speed=1.25),
)

# Two task types: latency-critical (small, urgent) and bulk (larger,
# tolerant of queueing). A task's "load" is how much capacity it
# consumes on whichever node it's assigned to until cleared.
TASK_TYPES = {
    "critical": {"load": 1, "deadline": 2},  # must clear within 2 steps
    "bulk": {"load": 3, "deadline": 6},
}


class EdgeSchedulingEnv(gym.Env):
    """
    Observation: for each node, (current_load / capacity) -- a
    continuous vector in [0, 1]^N -- plus a one-hot encoding of the
    incoming task's type. Action: which node (0..N-1) to assign the
    incoming task to. An additional "reject" action (index N) is
    always available, since sending a task to an already-overloaded
    node can be worse than dropping it.

    Reward per step:
      +1.0  if the task is accepted onto a node with enough free
            capacity, scaled by how well the task type's urgency
            matches the node's speed (a critical task on a fast node
            with room scores higher than the same task on a slow,
            nearly-full node)
      -1.0  if the task is rejected (dropped) OR assigned to a node
            that doesn't actually have capacity (an invalid/forced
            drop), tasks then wait in the meantime
      Every step, existing load on every node is cleared according to
      that node's speed, so nodes recover capacity over time -- an
      agent that dumps everything on one node will see it fill up and
      start forcing drops a few steps later.
    """

    def __init__(self, nodes=DEFAULT_NODES, max_steps: int = 200, seed: Optional[int] = None):
        super().__init__()
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        # action space: choose a node (0..n_nodes-1) or reject (n_nodes)
        self.action_space = spaces.Discrete(self.n_nodes + 1)
        # observation: n_nodes load-fractions + 2 one-hot task-type bits
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_nodes + 2,), dtype=np.float32
        )

        self._loads = np.zeros(self.n_nodes, dtype=np.float64)
        self._step_count = 0
        self._current_task_type = None
        self._dropped_count = 0
        self._accepted_count = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._loads = np.zeros(self.n_nodes, dtype=np.float64)
        self._step_count = 0
        self._dropped_count = 0
        self._accepted_count = 0
        self._current_task_type = self._sample_task_type()
        return self._observe(), {}

    def _sample_task_type(self) -> str:
        # Critical tasks are the minority but disproportionately matter
        # for reward -- mirrors a realistic industrial mix (most
        # traffic is bulk telemetry/logging, a minority is
        # control-loop-critical).
        return "critical" if self._rng.random() < 0.3 else "bulk"

    def _observe(self) -> np.ndarray:
        load_frac = np.array(
            [self._loads[i] / self.nodes[i].capacity for i in range(self.n_nodes)],
            dtype=np.float32,
        )
        task_onehot = np.array(
            [1.0 if self._current_task_type == "critical" else 0.0,
             1.0 if self._current_task_type == "bulk" else 0.0],
            dtype=np.float32,
        )
        return np.concatenate([load_frac, task_onehot])

    def step(self, action: int):
        assert self.action_space.contains(action), f"invalid action {action}"
        task = TASK_TYPES[self._current_task_type]
        reward = 0.0
        info = {}

        if action == self.n_nodes:
            # explicit reject
            reward = -1.0
            self._dropped_count += 1
            info["outcome"] = "rejected"
        else:
            node = self.nodes[action]
            free_capacity = node.capacity - self._loads[action]
            if free_capacity >= task["load"]:
                self._loads[action] += task["load"]
                self._accepted_count += 1
                # reward shaping: reward is higher when a critical task
                # lands on a comparatively fast node, and lower (but
                # still positive) when a bulk task lands on a slow
                # node -- both are "fine", but the agent should prefer
                # to save fast-node capacity for critical tasks when
                # it has a choice.
                speed_bonus = 0.3 * (node.speed / max(n.speed for n in self.nodes))
                if self._current_task_type == "critical":
                    reward = 1.0 + speed_bonus
                else:
                    reward = 1.0 - 0.2 * speed_bonus
                info["outcome"] = "accepted"
            else:
                # forced drop: agent picked a node with no room
                reward = -1.0
                self._dropped_count += 1
                info["outcome"] = "forced_drop"

        # clear load on every node according to its speed (independent
        # of this step's action) -- this is what makes "just always
        # pick the biggest node" a bad long-run policy: the biggest
        # node here is also the slowest, so load piles up on it.
        for i, node in enumerate(self.nodes):
            self._loads[i] = max(0.0, self._loads[i] - node.speed)

        self._current_task_type = self._sample_task_type()
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps
        return self._observe(), reward, terminated, truncated, info

    def summary(self) -> dict:
        total = self._accepted_count + self._dropped_count
        return {
            "accepted": self._accepted_count,
            "dropped": self._dropped_count,
            "total": total,
            "accept_rate": self._accepted_count / total if total else 0.0,
        }
