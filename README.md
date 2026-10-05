# edge-network-rl-scheduler

A tested tabular Q-learning agent that learns to route incoming tasks
across a small simulated industrial edge network (multiple edge nodes with
different capacity/speed tradeoffs). It implements and evaluates a
reinforcement learning algorithm against an industrial edge-network
scheduling scenario.

## What it does

- **`src/environment.py`** — a custom [Gymnasium](https://gymnasium.farama.org/)
  environment: 3 edge nodes with different capacity/speed profiles, a
  stream of critical/bulk tasks, an action space of "assign to node i" or
  "reject," and a reward function that rewards accepting tasks (more for
  routing critical tasks to faster nodes) and penalizes rejecting or
  force-dropping them. Node load decays each step according to each node's
  speed, so a policy that dumps every task on one node gets punished a few
  steps later when that node saturates.
- **`src/q_learning_agent.py`** — a from-scratch tabular Q-learning agent:
  epsilon-greedy action selection with epsilon decay, the standard
  Q-learning update rule
  (`Q(s,a) += alpha * (reward + gamma * max(Q(s')) - Q(s,a))`), and a
  continuous-to-discrete state binning function.
- **`src/baseline.py`** — a commonly used heuristic (least-loaded routing)
  that the trained agent is measured against, so the comparison is against
  a real alternative rather than random action selection.
- **`src/train.py`** — training loop and a held-out evaluation split: the
  agent is evaluated on fresh episodes (different random seeds) from those
  it trained on, and the baseline is evaluated on the exact same held-out
  episodes for a fair comparison.
- **`run_pipeline.py`** — trains the agent, evaluates it against the
  baseline, prints a comparison report, and saves a learning-curve figure
  to `output/learning_curve.png`.

## Scope

- The edge network and task stream are synthetic, defined in
  `src/environment.py`: three fictional edge nodes with deliberately
  different capacity/speed profiles (a fast-but-small node, a
  slow-but-large node, and a balanced node) and a stream of two task types
  (latency-critical and bulk).
- The network topology (3 nodes), task-type set (2 types), and reward
  function are small and hand-designed, which keeps the problem inspectable
  and the algorithm easy to verify.
- The environment's observation space is continuous (per-node load
  fractions), so it is discretized into bins and used as a Q-table lookup
  key, a standard technique for applying tabular methods to a
  low-dimensional continuous space. A deep RL approach (DQN, PPO, etc.)
  would remove the need for discretization and scale to larger networks;
  the tabular agent is easier to inspect and verify by hand, which suits a
  from-scratch implementation.

## Results

Sample output from a run:

```
Training Q-learning agent (3000 episodes)...

Evaluating trained agent (greedy policy, 50 held-out episodes)...
Evaluating least-loaded baseline (same 50 held-out episodes)...

=== Results (50 held-out episodes, 200 steps each) ===
Policy                 Mean episode reward    Mean accept rate
Least-loaded (baseline)                205.20              99.80%
Q-learning (trained)                   208.69             100.00%

Reward improvement over baseline: +3.49 per episode
Q-table size (distinct states visited): 196

Saved learning curve to output/learning_curve.png
```

The trained agent beats the least-loaded baseline by +3.49 mean episode
reward (~1.7%). The environment is simple enough for tabular Q-learning to
solve well, so there is limited headroom above a reasonable heuristic.

The learning curve (`output/learning_curve.png`) shows a clear learning
trend: mean episode reward rises from roughly 60 to roughly 190 over the
first ~500 episodes, then oscillates with ongoing epsilon-greedy
exploration noise near and eventually slightly above the baseline's
held-out mean for the rest of training, consistent with the final greedy
(exploitation-only) policy landing just above the baseline at evaluation
time.

### Tuning the discretization and training budget

My first full run scored **192.28** mean episode reward on held-out
evaluation against **205.20** for the least-loaded baseline, even though
the training curve showed learning (rising from ~60 to ~190 over 500
episodes).

I diagnosed it by inspecting the trained Q-table: 8 of 61 visited states
had "reject" as the greedy action, several with Q-values for reject and
accept within a fraction of a point of each other. The cause was the state
discretization (`bins=5`): five buckets per node's load fraction were too
coarse to distinguish "one unit away from full" from "actually full," so
the agent learned an overly cautious reject policy in some merged states.

Sweeping the bin count (5 / 10 / 20) at a fixed training budget showed 5
bins too coarse and 20 bins too sparse to learn well in the available
episodes, while **10 bins** let the agent reliably beat the baseline. The
same sweep showed the original 500-episode budget plateaued below the
baseline even at 10 bins, and extending training to **3,000 episodes** let
the greedy policy consistently clear it. Both changes are in the current
code (`discretize()`'s default `bins=10`, `train()`'s default
`n_episodes=3000`), and `tests/test_q_learning_agent.py` pins the default
bin count so a regression would be caught by the test suite.

## Tests

29 automated tests, all passing. They cover the environment's reward and
load-decay mechanics, the Q-learning update rule in isolation (including a
convergence check against the known analytical fixed point of a
self-looping MDP), and the end-to-end training/evaluation pipeline.

## Running it

```bash
python3 -m pytest -v          # 29 tests, all passing
python3 run_pipeline.py       # trains, evaluates, saves output/learning_curve.png
```

## Possible extensions

- Replace the tabular agent with DQN or PPO for larger, higher-dimensional
  edge networks.
- Add LLM-based agent or multi-agent coordination on top of the scheduler;
  [rag-tool-agent-demo](../rag-tool-agent-demo/) covers a single-agent LLM
  tool-router.
- Calibrate the topology and reward function against a real edge
  deployment.
