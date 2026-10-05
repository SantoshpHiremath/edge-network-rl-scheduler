"""
A simple heuristic baseline policy, used to give the learned Q-learning
agent a solid reference to be measured against -- "the agent learned
something" only means something if it's compared to a real alternative,
not just to random action selection.

Heuristic: always assign the task to whichever node currently has the
most free capacity (a standard "least-loaded" load-balancing rule),
falling back to reject if no node has room. This is a genuinely
reasonable, commonly-used real-world heuristic, not a strawman -- so
the agent has to actually learn a better *speed-aware* policy to beat
it, not just outperform a deliberately bad baseline.
"""

from src.environment import EdgeSchedulingEnv, TASK_TYPES


def least_loaded_action(env: EdgeSchedulingEnv) -> int:
    free = [env.nodes[i].capacity - env._loads[i] for i in range(env.n_nodes)]
    best_node = max(range(env.n_nodes), key=lambda i: free[i])
    task_load = TASK_TYPES[env._current_task_type]["load"]
    if free[best_node] >= task_load:
        return best_node
    return env.n_nodes  # reject
