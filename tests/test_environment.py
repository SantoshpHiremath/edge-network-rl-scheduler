import numpy as np
import pytest

from src.environment import EdgeSchedulingEnv, EdgeNode, TASK_TYPES


def make_env(seed=0, **kwargs):
    return EdgeSchedulingEnv(seed=seed, **kwargs)


def test_reset_returns_zero_load_observation():
    env = make_env()
    obs, info = env.reset(seed=0)
    n_nodes = env.n_nodes
    assert np.allclose(obs[:n_nodes], 0.0)
    assert info == {}


def test_observation_shape_matches_space():
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


def test_accepting_a_task_increases_node_load_before_decay():
    env = make_env(seed=1)
    env.reset(seed=1)
    # force the task type deterministically for this test
    env._current_task_type = "bulk"
    node_idx = 1  # edge-slow-large, capacity=8, speed=0.75
    obs, reward, terminated, truncated, info = env.step(node_idx)
    assert info["outcome"] == "accepted"
    assert reward > 0
    # load after this step = task_load(3) - node_speed(0.75) = 2.25
    assert env._loads[node_idx] == pytest.approx(2.25)


def test_rejecting_a_task_gives_negative_reward():
    env = make_env(seed=2)
    env.reset(seed=2)
    obs, reward, terminated, truncated, info = env.step(env.n_nodes)  # reject action
    assert reward == -1.0
    assert info["outcome"] == "rejected"


def test_forced_drop_when_node_has_no_capacity():
    env = make_env(seed=3, max_steps=1000)
    env.reset(seed=3)
    # Manually saturate node 0 (capacity=3) past what a single task
    # could fit, then force-assign a bulk task (load=3) to it.
    env._loads[0] = 3.0
    env._current_task_type = "bulk"
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["outcome"] == "forced_drop"
    assert reward == -1.0


def test_load_decays_by_node_speed_each_step_even_on_reject():
    env = make_env(seed=4)
    env.reset(seed=4)
    env._loads[:] = [2.0, 5.0, 3.0]
    expected_after = [
        max(0.0, 2.0 - env.nodes[0].speed),
        max(0.0, 5.0 - env.nodes[1].speed),
        max(0.0, 3.0 - env.nodes[2].speed),
    ]
    env.step(env.n_nodes)  # reject -- decay still happens
    assert list(env._loads) == pytest.approx(expected_after)


def test_load_never_goes_negative_after_decay():
    env = make_env(seed=5)
    env.reset(seed=5)
    env._loads[:] = [0.1, 0.1, 0.1]
    env.step(env.n_nodes)
    assert all(load >= 0.0 for load in env._loads)


def test_critical_task_on_fastest_node_scores_higher_than_on_slowest():
    """Same task type, same free capacity available on both nodes --
    the reward should reward routing critical tasks to the faster
    node, since that's the whole point of a speed-aware policy beating
    a naive least-loaded heuristic."""
    fastest_idx = max(range(3), key=lambda i: EdgeSchedulingEnv().nodes[i].speed)
    slowest_idx = min(range(3), key=lambda i: EdgeSchedulingEnv().nodes[i].speed)

    env_fast = make_env(seed=6)
    env_fast.reset(seed=6)
    env_fast._current_task_type = "critical"
    _, reward_fast, *_ = env_fast.step(fastest_idx)

    env_slow = make_env(seed=6)
    env_slow.reset(seed=6)
    env_slow._current_task_type = "critical"
    _, reward_slow, *_ = env_slow.step(slowest_idx)

    assert reward_fast > reward_slow


def test_episode_truncates_at_max_steps():
    env = make_env(seed=7, max_steps=5)
    env.reset(seed=7)
    truncated = False
    steps = 0
    while not truncated and steps < 100:
        _, _, terminated, truncated, _ = env.step(env.n_nodes)
        steps += 1
    assert steps == 5
    assert truncated is True


def test_summary_counts_accepted_and_dropped_correctly():
    env = make_env(seed=8, max_steps=1000)
    env.reset(seed=8)
    env.step(env.n_nodes)  # reject -> dropped
    env._current_task_type = "critical"
    env.step(0)  # should be accepted (node has room)
    summary = env.summary()
    assert summary["accepted"] == 1
    assert summary["dropped"] == 1
    assert summary["total"] == 2
    assert summary["accept_rate"] == pytest.approx(0.5)


def test_invalid_action_raises():
    env = make_env(seed=9)
    env.reset(seed=9)
    with pytest.raises(AssertionError):
        env.step(env.n_nodes + 1)  # out of range


def test_task_type_distribution_roughly_matches_configured_probability():
    """Sanity check the sampling isn't broken (e.g. always returning
    the same type) -- not a strict statistical test, just a bounds
    check with a fixed seed for reproducibility."""
    env = make_env(seed=123, max_steps=2000)
    env.reset(seed=123)
    critical_count = 0
    total = 500
    for _ in range(total):
        if env._current_task_type == "critical":
            critical_count += 1
        env.step(env.n_nodes)
    fraction = critical_count / total
    assert 0.20 < fraction < 0.40  # configured probability is 0.3
