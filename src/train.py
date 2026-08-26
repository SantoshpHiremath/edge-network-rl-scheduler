"""
Training and evaluation harness: trains the Q-learning agent over many
episodes, then evaluates the trained (greedy, epsilon=0) policy against
the least-loaded baseline over held-out episodes with different random
seeds than training used -- a genuine train/evaluate split for a
sequential-decision problem, not testing on the same episodes it
trained on.
"""

from typing import Optional

import numpy as np

from src.environment import EdgeSchedulingEnv
from src.q_learning_agent import QLearningAgent, discretize
from src.baseline import least_loaded_action


def train(
    n_episodes: int = 3000,
    max_steps: int = 200,
    seed: int = 42,
) -> QLearningAgent:
    env = EdgeSchedulingEnv(max_steps=max_steps, seed=seed)
    agent = QLearningAgent(n_actions=env.action_space.n, seed=seed)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 10_000 + ep)
        state_key = discretize(obs)
        done = False
        while not done:
            action = agent.select_action(state_key)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state_key = discretize(next_obs)
            done = terminated or truncated
            agent.update(state_key, action, reward, next_state_key, done)
            state_key = next_state_key
        agent.decay_epsilon()

    return agent


def evaluate_agent(agent: QLearningAgent, n_episodes: int = 50,
                    max_steps: int = 200, seed: int = 999) -> dict:
    """Evaluate the trained agent greedily (no exploration) on fresh
    episodes the agent never trained on."""
    env = EdgeSchedulingEnv(max_steps=max_steps, seed=seed)
    total_reward = 0.0
    total_accept_rate = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 10_000 + ep)
        state_key = discretize(obs)
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(state_key, greedy=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            state_key = discretize(next_obs)
        total_reward += ep_reward
        total_accept_rate += env.summary()["accept_rate"]
    return {
        "mean_episode_reward": total_reward / n_episodes,
        "mean_accept_rate": total_accept_rate / n_episodes,
    }


def evaluate_baseline(n_episodes: int = 50, max_steps: int = 200,
                       seed: int = 999) -> dict:
    """Evaluate the least-loaded heuristic on the exact same held-out
    episodes (same seeds) used to evaluate the trained agent, so the
    comparison is apples-to-apples."""
    total_reward = 0.0
    total_accept_rate = 0.0
    for ep in range(n_episodes):
        env = EdgeSchedulingEnv(max_steps=max_steps, seed=seed)
        obs, _ = env.reset(seed=seed * 10_000 + ep)
        done = False
        ep_reward = 0.0
        while not done:
            action = least_loaded_action(env)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_reward += ep_reward
        total_accept_rate += env.summary()["accept_rate"]
    return {
        "mean_episode_reward": total_reward / n_episodes,
        "mean_accept_rate": total_accept_rate / n_episodes,
    }
