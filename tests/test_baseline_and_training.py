import pytest

from src.environment import EdgeSchedulingEnv
from src.baseline import least_loaded_action
from src.train import train, evaluate_agent, evaluate_baseline


def test_baseline_picks_node_with_most_free_capacity():
    env = EdgeSchedulingEnv(seed=0)
    env.reset(seed=0)
    env._loads[:] = [2.5, 6.0, 1.0]  # free: 0.5, 2.0, 4.0
    env._current_task_type = "bulk"  # load=3
    action = least_loaded_action(env)
    assert action == 2  # node 2 has the most free capacity (4.0) and fits load=3


def test_baseline_rejects_when_no_node_has_room():
    env = EdgeSchedulingEnv(seed=1)
    env.reset(seed=1)
    env._loads[:] = [3.0, 8.0, 5.0]  # all nodes fully saturated
    env._current_task_type = "critical"  # load=1, still no room anywhere
    action = least_loaded_action(env)
    assert action == env.n_nodes  # reject


def test_baseline_evaluation_produces_finite_results():
    results = evaluate_baseline(n_episodes=5, max_steps=50, seed=42)
    assert "mean_episode_reward" in results
    assert "mean_accept_rate" in results
    assert 0.0 <= results["mean_accept_rate"] <= 1.0


def test_training_produces_a_populated_q_table():
    agent = train(n_episodes=30, max_steps=50, seed=0)
    assert len(agent.q_table) > 0
    # after training, epsilon should have decayed from its starting value
    assert agent.epsilon < 1.0


def test_trained_agent_evaluation_returns_finite_results():
    agent = train(n_episodes=30, max_steps=50, seed=0)
    results = evaluate_agent(agent, n_episodes=5, max_steps=50, seed=999)
    assert "mean_episode_reward" in results
    assert 0.0 <= results["mean_accept_rate"] <= 1.0


def test_trained_agent_outperforms_a_freshly_initialized_agent():
    """A trained agent should do meaningfully better than an
    equivalent but completely untrained agent evaluated greedily (all
    Q-values are zero, so it always picks action 0 -- a fixed, naive
    policy). This isolates the effect of training itself, independent
    of the baseline heuristic comparison."""
    trained_agent = train(n_episodes=300, max_steps=100, seed=0)
    trained_results = evaluate_agent(trained_agent, n_episodes=30, max_steps=100, seed=999)

    from src.q_learning_agent import QLearningAgent
    untrained_agent = QLearningAgent(n_actions=trained_agent.n_actions, seed=0)
    untrained_results = evaluate_agent(untrained_agent, n_episodes=30, max_steps=100, seed=999)

    assert trained_results["mean_episode_reward"] > untrained_results["mean_episode_reward"]
