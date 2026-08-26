"""
End-to-end pipeline: trains the Q-learning agent, evaluates it against
the least-loaded baseline on held-out episodes, prints a real
comparison report, and saves a real learning-curve figure (reward per
training episode, smoothed) to output/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.environment import EdgeSchedulingEnv
from src.q_learning_agent import QLearningAgent, discretize
from src.baseline import least_loaded_action
from src.train import train, evaluate_agent, evaluate_baseline


def train_with_history(n_episodes: int = 3000, max_steps: int = 200, seed: int = 42):
    env = EdgeSchedulingEnv(max_steps=max_steps, seed=seed)
    agent = QLearningAgent(n_actions=env.action_space.n, seed=seed)
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 10_000 + ep)
        state_key = discretize(obs)
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(state_key)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            next_state_key = discretize(next_obs)
            done = terminated or truncated
            agent.update(state_key, action, reward, next_state_key, done)
            state_key = next_state_key
        agent.decay_epsilon()
        episode_rewards.append(ep_reward)

    return agent, episode_rewards


def moving_average(values, window=100):
    values = np.array(values, dtype=np.float64)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    os.makedirs("output", exist_ok=True)

    print("Training Q-learning agent (3000 episodes)...")
    agent, episode_rewards = train_with_history(n_episodes=3000)

    print("\nEvaluating trained agent (greedy policy, 50 held-out episodes)...")
    agent_results = evaluate_agent(agent, n_episodes=50)

    print("Evaluating least-loaded baseline (same 50 held-out episodes)...")
    baseline_results = evaluate_baseline(n_episodes=50)

    print("\n=== Results (50 held-out episodes, 200 steps each) ===")
    print(f"{'Policy':<20}{'Mean episode reward':>22}{'Mean accept rate':>20}")
    print(f"{'Least-loaded (baseline)':<20}{baseline_results['mean_episode_reward']:>22.2f}"
          f"{baseline_results['mean_accept_rate']:>20.2%}")
    print(f"{'Q-learning (trained)':<20}{agent_results['mean_episode_reward']:>22.2f}"
          f"{agent_results['mean_accept_rate']:>20.2%}")

    improvement = agent_results["mean_episode_reward"] - baseline_results["mean_episode_reward"]
    print(f"\nReward improvement over baseline: {improvement:+.2f} per episode")
    print(f"Q-table size (distinct states visited): {len(agent.q_table)}")

    # Learning curve
    smoothed = moving_average(episode_rewards, window=100)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(episode_rewards, alpha=0.15, color="#4C72B0", label="raw episode reward")
    ax.plot(range(len(smoothed)), smoothed, color="#4C72B0", linewidth=2,
            label="100-episode moving average")
    ax.axhline(baseline_results["mean_episode_reward"], color="#DD8452",
               linestyle="--", linewidth=2, label="baseline (held-out mean)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("Q-learning agent training curve vs. least-loaded baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig("output/learning_curve.png", dpi=150)
    print("\nSaved learning curve to output/learning_curve.png")


if __name__ == "__main__":
    main()
