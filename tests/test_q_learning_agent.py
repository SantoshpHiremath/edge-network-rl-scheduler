import numpy as np
import pytest

from src.q_learning_agent import QLearningAgent, discretize


def test_discretize_binning_boundaries():
    obs = np.array([0.0, 0.5, 0.999, 1.0, 0.0], dtype=np.float32)
    # last two entries are the one-hot task type -- treat first 3 as
    # load fractions for a 3-node env
    key = discretize(obs, bins=5)
    assert key == (0, 2, 4, 1, 0)  # 1.0 clamps into the top bin (index 4), not 5


def test_discretize_default_bin_count_is_ten():
    """The default bin count was tuned empirically (see README):
    5 bins was too coarse to distinguish 'nearly full' from 'full'
    and caused the trained agent to underperform the baseline; 10
    bins was the setting that let the agent reliably beat it."""
    obs = np.array([0.0, 0.5, 0.999, 1.0, 0.0], dtype=np.float32)
    key = discretize(obs)  # uses the default bin count
    assert key == (0, 5, 9, 1, 0)


def test_discretize_is_deterministic_for_same_input():
    obs = np.array([0.3, 0.7, 0.1, 0.0, 1.0], dtype=np.float32)
    assert discretize(obs) == discretize(obs)


def test_agent_initializes_q_table_lazily_with_zeros():
    agent = QLearningAgent(n_actions=4, seed=0)
    state = (0, 0, 0, 1, 0)
    assert state not in agent.q_table
    q_values = agent.q_table[state]
    assert np.array_equal(q_values, np.zeros(4))


def test_select_action_returns_valid_action_index():
    agent = QLearningAgent(n_actions=4, seed=1)
    state = (1, 2, 3, 0, 1)
    for _ in range(20):
        action = agent.select_action(state)
        assert 0 <= action < 4


def test_greedy_selection_picks_highest_q_value():
    agent = QLearningAgent(n_actions=4, seed=2)
    state = (0, 0, 0, 0, 0)
    agent.q_table[state] = np.array([0.1, 0.9, 0.3, 0.2])
    # greedy=True must never explore randomly
    for _ in range(20):
        action = agent.select_action(state, greedy=True)
        assert action == 1


def test_update_moves_q_value_toward_target_not_past_it():
    agent = QLearningAgent(n_actions=2, alpha=0.5, gamma=0.9, seed=3)
    state = (0, 0)
    next_state = (1, 1)
    agent.q_table[state] = np.array([0.0, 0.0])
    agent.q_table[next_state] = np.array([2.0, 0.0])  # max future Q = 2.0

    agent.update(state, action=0, reward=1.0, next_state_key=next_state, done=False)
    # target = 1.0 + 0.9*2.0 = 2.8; new_q = 0 + 0.5*(2.8-0) = 1.4
    assert agent.q_table[state][0] == pytest.approx(1.4)


def test_update_with_done_ignores_future_reward():
    agent = QLearningAgent(n_actions=2, alpha=0.5, gamma=0.9, seed=4)
    state = (0, 0)
    next_state = (1, 1)
    agent.q_table[state] = np.array([0.0, 0.0])
    agent.q_table[next_state] = np.array([100.0, 0.0])  # should be ignored

    agent.update(state, action=0, reward=1.0, next_state_key=next_state, done=True)
    # target = reward only = 1.0; new_q = 0 + 0.5*(1.0-0) = 0.5
    assert agent.q_table[state][0] == pytest.approx(0.5)


def test_repeated_updates_converge_toward_true_value_for_deterministic_reward():
    """A simple sanity/convergence check: repeatedly updating the same
    state-action pair with the same reward should converge the Q-value
    toward reward / (1 - gamma) for a self-looping deterministic MDP,
    confirming the update rule is mathematically correct, not just
    'runs without crashing'."""
    agent = QLearningAgent(n_actions=1, alpha=0.3, gamma=0.9, seed=5)
    state = (0,)
    for _ in range(500):
        agent.update(state, action=0, reward=1.0, next_state_key=state, done=False)
    expected_fixed_point = 1.0 / (1 - 0.9)  # = 10.0
    assert agent.q_table[state][0] == pytest.approx(expected_fixed_point, rel=0.05)


def test_epsilon_decays_but_does_not_go_below_floor():
    agent = QLearningAgent(n_actions=2, epsilon_start=1.0, epsilon_end=0.1,
                            epsilon_decay=0.9, seed=6)
    for _ in range(200):
        agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.1)
    assert agent.epsilon >= 0.1


def test_tie_breaking_is_random_not_always_first_index():
    """With an all-zero Q-row (a fresh/unvisited state), action
    selection under greedy mode should not always return action 0 --
    otherwise the agent would have a silent bias toward
    low-indexed actions before it has learned anything."""
    agent = QLearningAgent(n_actions=5, seed=7)
    state = (9, 9, 9, 9, 9)  # never touched -> all zeros
    seen_actions = set()
    for _ in range(100):
        seen_actions.add(agent.select_action(state, greedy=True))
    assert len(seen_actions) > 1
