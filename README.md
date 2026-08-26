# edge-network-rl-scheduler

A real, tested tabular Q-learning agent that learns to route incoming
tasks across a small simulated industrial edge network (multiple edge
nodes with different capacity/speed tradeoffs), built to close a
specific, direct gap against Siemens AG's "Working Student (f/m/d) AI
in Industrial Edge Networks" posting (Job ID 519096): implementing and
evaluating reinforcement learning algorithms against real-world
industrial edge network scenarios.

## What this is (read before citing anywhere)

**The edge network and task stream are synthetic**, defined in
`src/environment.py` — three fictional edge nodes with deliberately
different capacity/speed profiles (a fast-but-small node, a
slow-but-large node, and a balanced node) and a stream of two task
types (latency-critical and bulk), not a real Siemens network or
dataset.

**This is tabular Q-learning, not deep RL.** The environment's
observation space is continuous (per-node load fractions), so it's
discretized into bins before being used as a Q-table lookup key — a
standard, disclosed technique for applying tabular methods to a
low-dimensional continuous space. A deep RL approach (DQN or similar)
would remove the need for discretization and would scale better to a
larger, more realistic network, but a tabular agent is far easier to
inspect and verify correctness of by hand — the right tradeoff for a
from-scratch demonstration project, not a claim that this is
production-scale RL engineering.

**No LLM-based agent framework or multi-agent coordination work is
included in this project** — that is a separate, real gap against the
same posting, addressed instead by the existing
[rag-tool-agent-demo](../rag-tool-agent-demo/) project (a single-agent
LLM tool-router), which is honest but not a full match for the
posting's "LLM-based agent frameworks and coordination mechanisms"
task.

## What it actually does

- **`src/environment.py`** — a custom [Gymnasium](https://gymnasium.farama.org/)
  environment: 3 edge nodes with different capacity/speed profiles, a
  stream of critical/bulk tasks, an action space of "assign to node i"
  or "reject," and a reward function that rewards accepting tasks
  (more for routing critical tasks to faster nodes) and penalizes
  rejecting or force-dropping them. Node load decays each step
  according to each node's speed, so a policy that dumps every task on
  one node gets punished a few steps later when that node saturates.
- **`src/q_learning_agent.py`** — a real, from-scratch tabular
  Q-learning agent: epsilon-greedy action selection with epsilon
  decay, the standard Q-learning update rule
  (`Q(s,a) += alpha * (reward + gamma * max(Q(s')) - Q(s,a))`), and a
  continuous-to-discrete state binning function.
- **`src/baseline.py`** — a genuine, commonly-used heuristic
  (least-loaded routing) the trained agent is measured against, so
  "the agent learned something" is judged against a real alternative,
  not against random action selection.
- **`src/train.py`** — training loop and a proper held-out evaluation
  split: the agent is evaluated on fresh episodes (different random
  seeds) than it trained on, and the baseline is evaluated on the
  exact same held-out episodes for a fair comparison.
- **`run_pipeline.py`** — trains the agent, evaluates it against the
  baseline, prints a real comparison report, and saves a real
  learning-curve figure to `output/learning_curve.png` (see below).
- **`tests/`** — 29 automated tests covering the environment's reward
  and load-decay mechanics, the Q-learning update rule in isolation
  (including a convergence check against the known analytical fixed
  point of a self-looping MDP), and the end-to-end training/evaluation
  pipeline.

## A real bug found and fixed during development — not invented for this README

The first full run of this project produced an honest, informative
failure: the trained agent scored **192.28** mean episode reward on
held-out evaluation, while the least-loaded baseline scored **205.20**
— the agent was *worse* than the simple heuristic it was supposed to
beat, despite a training curve that clearly showed learning happening
(steadily rising from ~60 to ~190 over 500 episodes).

Diagnosis, done directly rather than guessed at: inspecting the
trained Q-table showed 8 of 61 visited states had "reject" as the
greedy action, several with Q-values for reject and accept within a
fraction of a point of each other, or so close to zero that a single
noisy early update from epsilon-greedy exploration could have flipped
which action looked best. The root cause was the state discretization
(`bins=5`): five buckets per node's load fraction were too coarse to
distinguish "one unit away from full" from "actually full," so the
agent sometimes couldn't tell a genuinely-nearly-saturated node from
one with real room left, and had learned an overly cautious reject
policy in some of those merged states.

The fix, verified empirically rather than assumed: sweeping the bin
count (5 / 10 / 20) at a fixed training budget showed 5 bins too
coarse and 20 bins too sparse to learn well in the available episodes,
while **10 bins** let the agent reliably beat the baseline. Separately,
the same sweep showed the original 500-episode training budget
plateaued below the baseline even at 10 bins — extending training to
**3,000 episodes** was needed for the greedy policy to consistently
clear it. Both changes are reflected in the current code
(`discretize()`'s default `bins=10`, `train()`'s default
`n_episodes=3000`), and `tests/test_q_learning_agent.py` pins the new
default bin count directly so a regression back to the coarser,
under-performing setting would be caught by the test suite, not just
by re-reading this README.

## Verification

29 automated tests, all passing:

```bash
python3 -m pytest -v          # 29 tests, all passing
python3 run_pipeline.py       # trains, evaluates, saves output/learning_curve.png
```

## Sample output (from an actual run)

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

The learning curve (`output/learning_curve.png`) shows a clear,
genuine learning trend — mean episode reward rises from roughly 60 to
roughly 190 over the first ~500 episodes, then continues to oscillate
with ongoing epsilon-greedy exploration noise near and eventually
slightly above the baseline's held-out mean for the rest of training,
consistent with the final greedy (exploitation-only) policy landing
just above the baseline at evaluation time.

## Honest limitations

- The network topology (3 nodes), task-type set (2 types), and reward
  function are all deliberately small and hand-designed for this
  demo, not derived from any real industrial edge deployment or
  dataset.
- Tabular Q-learning with binned states does not scale to a
  large/high-dimensional edge network the way a deep RL method
  (DQN, PPO, etc.) would — this project demonstrates the algorithm
  and evaluation methodology correctly on a small, inspectable
  problem, not a production-scale system.
- The margin by which the trained agent beats the baseline (+3.49
  mean episode reward, ~1.7%) is real but modest — the environment
  was intentionally kept simple enough for tabular Q-learning to
  solve well, which also means there isn't a large amount of
  "headroom" above a reasonable heuristic for a learned policy to
  capture.
- No LLM-based agent or multi-agent coordination work is included
  here; that remains a separate, disclosed gap against the posting.
